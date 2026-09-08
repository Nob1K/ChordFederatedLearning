import sys
import os
import time
import threading
import hashlib

sys.path.append('gen-py')

import thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from compute import compute
from compute.ttypes import node, weights
from ML import ML
from config import (RING_SIZE, FINGER_TABLE_SIZE, SUCCESSOR_LIST_SIZE,
                    STABILIZE_INTERVAL, RPC_TIMEOUT,
                    NUM_CLASSES, HIDDEN_UNITS, LEARNING_RATE, TRAIN_EPOCHS, MOMENTUM)

# sentinel for "no node", since thrift can't send None. id == -1 means empty
NULL_NODE = node("", 0, -1)


# consistent hashing function used in the system
def hash_to_number(input_string):
    sha1_hash = hashlib.sha1(input_string.encode()).hexdigest()
    hash_int = int(sha1_hash, 16)

    return hash_int % RING_SIZE


# short id for log lines
def short_id(node_id):
    return f"{node_id:04x}"


# raised when an rpc to a peer fails, so the peer is treated as dead
class NodeDown(Exception):
    pass


class ComputeHandler:
    def __init__(self, host, port, bootstrap_host=None, bootstrap_port=None):
        self.ip = host
        self.port = port
        # chord id obtained by hashing own addr
        self.node_id = hash_to_number(f"{host}:{port}")
        # bootstrap peer or None for first node in ring
        self.bootstrap_host = bootstrap_host
        self.bootstrap_port = bootstrap_port

        self.lock = threading.RLock()
        self.model_lock = threading.RLock()

        self.predecessor = None
        # successor_list[0] is the immediate successor, the rest are backups
        self.successor_list = [self._self_node()]
        self.finger_table = [{} for _ in range(FINGER_TABLE_SIZE)]
        # next finger to fix
        self._next_finger = 0
        # last ring state we logged
        self._last_state = None

        self.models = {}
        # files being trained right now, so we don't start the same one twice
        self.training = set()
        self.train_lock = threading.RLock()
        # ML.init seeds numpy's global RNG, so serialize that part across threads
        self.init_lock = threading.Lock()

        self.join_network()

        print(f"Compute node initialized with IP: {self.ip}, Port: {port}, "
              f"ID: {self.node_id} ({short_id(self.node_id)})")
        self.print_info()

    # ----- helpers -------------------------------------------------------

    def _self_node(self):
        return node(self.ip, self.port, self.node_id)

    @property
    def successor(self):
        with self.lock:
            return self.successor_list[0]

    def _is_self(self, n):
        return n is not None and n.port == self.port and n.ip == self.ip

    """helper to check if id is in the range (start, end]."""
    def _is_between(self, id, start, end):
        if start < end:
            return start < id <= end
        elif start > end:
            return id > start or id <= end
        else:
            return id != start

    # ----- rpc with timeouts + failure detection ------------------------

    """open a client to a peer with a timeout"""
    def _open(self, n):
        transport = TSocket.TSocket(n.ip, n.port)
        transport.setTimeout(int(RPC_TIMEOUT * 1000))  # ms
        transport = TTransport.TBufferedTransport(transport)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        client = compute.Client(protocol)
        transport.open()
        return client, transport

    """call fn(client) on a peer, raise NodeDown if it fails"""
    def _call(self, n, fn):
        transport = None
        try:
            client, transport = self._open(n)
            return fn(client)
        except NodeDown:
            raise
        except Exception:
            raise NodeDown(f"{n.ip}:{n.port}")
        finally:
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass

    """drop a dead node from the finger table and successor list"""
    def _mark_dead(self, dead):
        with self.lock:
            for i in range(FINGER_TABLE_SIZE):
                ft = self.finger_table[i]
                if ft and ft["node"].id == dead.id:
                    ft["node"] = self._self_node()
                    ft["successor_id"] = self.node_id
            self.successor_list = [s for s in self.successor_list if s.id != dead.id]
            if not self.successor_list:
                self.successor_list = [self._self_node()]

    # ----- joining -------------------------------------------------------

    """join via a bootstrap peer, or form a new ring if none is given"""
    def join_network(self):
        self_node = self._self_node()
        # fingers start pointing at self, stabilization fixes them later
        for i in range(FINGER_TABLE_SIZE):
            self.finger_table[i] = {"start": (self.node_id + 2**i) % RING_SIZE,
                                    "successor_id": self.node_id,
                                    "node": self_node}

        # first node, form a ring of one
        if self.bootstrap_host is None:
            print("First node in the network, formed a new ring")
            self.predecessor = None
            self.successor_list = [self_node]
            return

        # build the seed's node locally (its id is just a hash of its address)
        bootstrap_id = hash_to_number(f"{self.bootstrap_host}:{self.bootstrap_port}")
        seed = node(self.bootstrap_host, self.bootstrap_port, bootstrap_id)
        print(f"Joining through bootstrap peer {seed.ip}:{seed.port} "
              f"(ID {seed.id} / {short_id(seed.id)})")

        # find our successor through the seed, retry if it isn't up yet
        succ = None
        for attempt in range(10):
            try:
                succ = self._call(seed, lambda c: c.find_successor(self.node_id))
                break
            except NodeDown:
                print(f"bootstrap peer not ready, retrying ({attempt + 1}/10)...")
                time.sleep(1)
        if succ is None:
            print("Error joining network: bootstrap peer unreachable")
            sys.exit(1)

        self.predecessor = None
        self.successor_list = [succ]
        self.finger_table[0]["node"] = succ
        self.finger_table[0]["successor_id"] = succ.id

    # ----- stabilization -------------------------------------------------

    """start the background maintenance loop"""
    def start_maintenance(self):
        t = threading.Thread(target=self._maintenance_loop, daemon=True)
        t.start()

    def _maintenance_loop(self):
        while True:
            try:
                self.stabilize()
                self._fix_one_finger()
                self.check_predecessor()
                self._log_state_on_change()
            except Exception as e:
                print(f"[maintenance] unexpected error: {e}")
            time.sleep(STABILIZE_INTERVAL)

    """check the successor, adopt a closer one if it appeared, refresh backups"""
    def stabilize(self):
        with self.lock:
            succ = self.successor_list[0]
            my_pred = self.predecessor

        if self._is_self(succ):
            # successor is self, so its predecessor is just our own predecessor.
            # this is how the first node picks up a successor.
            x = my_pred
            succ_backups = []
        else:
            try:
                x = self._call(succ, lambda c: c.get_predecessor())
                succ_backups = self._call(succ, lambda c: c.get_successor_list())
            except NodeDown:
                # successor dead, drop it and promote a backup
                print(f"[stabilize] successor {short_id(succ.id)} is down, promoting backup")
                self._mark_dead(succ)
                return
            if x is not None and x.id == -1:
                x = None

        # adopt x if it sits between us and succ (when succ is self the interval
        # is the whole ring, so any real x is taken)
        if x is not None and not self._is_self(x) and self._is_between(x.id, self.node_id, succ.id):
            succ = x
            try:
                succ_backups = self._call(succ, lambda c: c.get_successor_list())
            except NodeDown:
                self._mark_dead(succ)
                return

        # still alone
        if self._is_self(succ):
            with self.lock:
                self.successor_list = [self._self_node()]
            return

        # successor_list = [succ] + succ's successors, deduped and capped
        new_list = [succ]
        for s in succ_backups:
            if len(new_list) >= SUCCESSOR_LIST_SIZE:
                break
            if s.id != succ.id and s.id != self.node_id:
                new_list.append(s)
        with self.lock:
            self.successor_list = new_list[:SUCCESSOR_LIST_SIZE]
            self.finger_table[0]["node"] = succ
            self.finger_table[0]["successor_id"] = succ.id

        # let the successor know about us
        try:
            self._call(succ, lambda c: c.notify(self._self_node()))
        except NodeDown:
            self._mark_dead(succ)

    """fix one finger table entry per tick"""
    def _fix_one_finger(self):
        with self.lock:
            self._next_finger = (self._next_finger + 1) % FINGER_TABLE_SIZE
            i = self._next_finger
            start = self.finger_table[i]["start"]
        try:
            succ = self.find_successor(start)
        except NodeDown:
            return
        with self.lock:
            self.finger_table[i]["node"] = succ
            self.finger_table[i]["successor_id"] = succ.id

    """clear the predecessor if it has died"""
    def check_predecessor(self):
        with self.lock:
            pred = self.predecessor
        if pred is None:
            return
        try:
            self._call(pred, lambda c: c.ping())
        except NodeDown:
            print(f"[check_predecessor] predecessor {short_id(pred.id)} is down, clearing")
            with self.lock:
                if self.predecessor is not None and self.predecessor.id == pred.id:
                    self.predecessor = None

    def _ring_state_str(self):
        with self.lock:
            pred = short_id(self.predecessor.id) if self.predecessor else "None"
            sl = ",".join(short_id(s.id) for s in self.successor_list)
        return f"[ring] id={short_id(self.node_id)} pred={pred} succ_list=[{sl}]"

    def _log_state_on_change(self):
        state = self._ring_state_str()
        if state != self._last_state:
            print(state)
            self._last_state = state

    # ----- routing -------------------------------------------------------

    """find the successor node for a given ID."""
    def find_successor(self, id):
        succ = self.successor
        # 1 node
        if self._is_self(succ):
            return self._self_node()

        # successor owns id
        if self._is_between(id, self.node_id, succ.id):
            return succ

        # forward to closest preceding node, fall back to successor if it's dead
        next_node = self.closest_preceding_node(id)
        if next_node is None or self._is_self(next_node):
            next_node = succ
        try:
            return self._call(next_node, lambda c: c.find_successor(id))
        except NodeDown:
            self._mark_dead(next_node)
            try:
                return self._call(self.successor, lambda c: c.find_successor(id))
            except NodeDown:
                return self._self_node()

    """find the closest preceding node for a given hash."""
    def closest_preceding_node(self, hash):
        with self.lock:
            for i in range(FINGER_TABLE_SIZE - 1, -1, -1):
                finger_successor_id = self.finger_table[i]["successor_id"]
                if self._is_between(finger_successor_id, self.node_id, hash):
                    return self.finger_table[i]["node"]
        return None

    """return this node's predecessor"""
    def get_predecessor(self):
        with self.lock:
            return self.predecessor if self.predecessor else NULL_NODE

    """return a copy of the successor list"""
    def get_successor_list(self):
        with self.lock:
            return list(self.successor_list)

    """liveness probe"""
    def ping(self):
        return

    """set a new predecessor if notified"""
    def notify(self, new_node):
        with self.lock:
            if self.predecessor is None or self._is_between(new_node.id, self.predecessor.id, self.node_id):
                self.predecessor = new_node

    # ----- data / training ----------------------------------------------

    """true if this node owns key h"""
    def _owns(self, h):
        with self.lock:
            pred = self.predecessor
        # no predecessor yet, claim it (corrected once stabilize sets one)
        if pred is None:
            return True
        return self._is_between(h, pred.id, self.node_id)

    """next hop toward the owner of key h"""
    def _next_hop(self, h):
        finger = self.closest_preceding_node(h)
        if finger is None or self._is_self(finger):
            return self.successor
        return finger

    """train the file if we own it, else forward toward the owner"""
    def put_data(self, filename):
        h = hash_to_number(filename)
        print(f"put_data {filename} (hash {h}) at node {short_id(self.node_id)}")
        if self._owns(h):
            self._ensure_training(filename)
        else:
            target = self._next_hop(h)
            print(f"redirecting file to node {short_id(target.id)}")
            try:
                self._call(target, lambda c: c.put_data(filename))
            except NodeDown:
                print(f"put_data forward to {short_id(target.id)} failed (node down)")

    """return the model for a file, retraining it here if it went missing"""
    def get_model(self, filename):
        h = hash_to_number(filename)
        print(f"get_model {filename} at node {short_id(self.node_id)}, hash {h}")
        if self._owns(h):
            with self.model_lock:
                if filename in self.models:
                    return self.models[filename]
            # we own it but don't have it (owner died), retrain from the file
            self._ensure_training(filename)
            return weights(w=[[0.0]], v=[[0.0]], status=1)  # wait
        target = self._next_hop(h)
        print(f"redirecting request to node {short_id(target.id)}")
        try:
            return self._call(target, lambda c: c.get_model(filename))
        except NodeDown:
            print(f"get_model forward to {short_id(target.id)} failed (node down)")
            return weights(w=[[0.0]], v=[[0.0]], status=-1)  # error

    """store a model copy sent from another node"""
    def replicate_model(self, filename, w):
        with self.model_lock:
            self.models[filename] = w

    """start training a file in the background unless it's already going"""
    def _ensure_training(self, filename):
        with self.train_lock:
            if filename in self.training:
                return
            with self.model_lock:
                if filename in self.models:
                    return
            self.training.add(filename)
        threading.Thread(target=self._train_and_store, args=(filename,), daemon=True).start()

    """train a file, store the result, and copy it to the backups"""
    def _train_and_store(self, filename):
        try:
            model = ML.mlp()
            # init sets the shared numpy seed, so only one thread inits at a time
            with self.init_lock:
                ready = model.init_training_random(filename, NUM_CLASSES, HIDDEN_UNITS)
            if ready:
                model.set_momentum(MOMENTUM)
                print(f"training {filename} at node {short_id(self.node_id)}")
                model.train(LEARNING_RATE, TRAIN_EPOCHS)
            v, w = model.get_weights()
            trained = weights(w, v, 0)
            with self.model_lock:
                self.models[filename] = trained
            self._replicate(filename, trained)
        except Exception as e:
            print(f"training {filename} failed: {e}")
        finally:
            with self.train_lock:
                self.training.discard(filename)

    """copy a trained model to the backup successors"""
    def _replicate(self, filename, w):
        with self.lock:
            targets = list(self.successor_list)
        copied = 0
        for t in targets:
            if t.id == self.node_id:
                continue
            try:
                self._call(t, lambda c: c.replicate_model(filename, w))
                copied += 1
            except NodeDown:
                pass
        if copied:
            print(f"replicated {filename} to {copied} backup(s)")

    """Print information about this node"""
    def print_info(self):
        with self.lock:
            print("\n----- Node Information -----")
            print(f"Node ID: {self.node_id} ({short_id(self.node_id)})")
            print(f"IP:Port: {self.ip}:{self.port}")
            if self.predecessor:
                print(f"Predecessor: {self.predecessor.ip}:{self.predecessor.port} (ID: {short_id(self.predecessor.id)})")
            else:
                print("Predecessor: None")
            sl = ", ".join(f"{s.ip}:{s.port}({short_id(s.id)})" for s in self.successor_list)
            print(f"Successor list: {sl}")
            print("Stored models:")
            for file in self.models.keys():
                print(f"  - {file}")
            print("----------------------------\n")


def start_server(host, port, bootstrap_host=None, bootstrap_port=None):
    """Start the Thrift server for this compute node."""
    handler = ComputeHandler(host, port, bootstrap_host, bootstrap_port)
    processor = compute.Processor(handler)

    server_transport = TSocket.TServerSocket(port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TThreadedServer(processor, server_transport, tfactory, pfactory)

    # start maintenance before we begin serving
    handler.start_maintenance()

    print(f"Starting compute node server on port: {port}")
    server.serve()


if __name__ == "__main__":
    # first node:   python3 compute_server.py <host> <port>
    # joining node: python3 compute_server.py <host> <port> <bootstrap_host> <bootstrap_port>
    if len(sys.argv) < 3:
        print("usage: python3 compute_server.py <host> <port> [bootstrap_host bootstrap_port]")
        sys.exit(1)
    host = sys.argv[1]
    port = int(sys.argv[2])
    bootstrap_host = sys.argv[3] if len(sys.argv) > 3 else None
    bootstrap_port = int(sys.argv[4]) if len(sys.argv) > 4 else None
    start_server(host, port, bootstrap_host, bootstrap_port)
