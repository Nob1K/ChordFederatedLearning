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
from config import RING_SIZE, FINGER_TABLE_SIZE, NUM_CLASSES, HIDDEN_UNITS, LEARNING_RATE, TRAIN_EPOCHS, MOMENTUM

# consistent hashing function used in the system: maps any string into the
# Chord identifier space [0, RING_SIZE). Used both for node IDs (hash of
# "host:port") and for data key (hash of a filename)
def hash_to_number(input_string):
    sha1_hash = hashlib.sha1(input_string.encode()).hexdigest()
    hash_int = int(sha1_hash, 16)

    return hash_int % RING_SIZE

# short, human-readable ID for log lines (the full IDs are large)
def short_id(node_id):
    return f"{node_id:04x}"

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
        self.successor = None
        self.finger_table = [{} for _ in range(FINGER_TABLE_SIZE)]

        self.models = {}
        self.data_files = set()

        self.join_network()

        print(f"Compute node initialized with IP: {self.ip}, Port: {port}, ID: {self.node_id} ({short_id(self.node_id)})")
        self.print_info()


    """join the Chord DHT network via a bootstrap peer, or form a new ring."""
    def join_network(self):
        try:
            self_node = node(self.ip, self.port, self.node_id)

            # first node and form a ring of one
            if self.bootstrap_host is None:
                print("First node in the network")
                self.predecessor = None
                self.successor = self_node
                for i in range(FINGER_TABLE_SIZE):
                    self.finger_table[i] = {"start": (self.node_id + 2**i) % RING_SIZE,
                                            "successor_id": self.node_id,
                                            "node": self_node}
                print("Formed a new ring")
                return

            # bootstrap through a known peer
            bootstrap_id = hash_to_number(f"{self.bootstrap_host}:{self.bootstrap_port}")
            existing_node = node(self.bootstrap_host, self.bootstrap_port, bootstrap_id)
            print(f"Joining through bootstrap peer {existing_node.ip}:{existing_node.port} "
                  f"(ID {existing_node.id} / {short_id(existing_node.id)})")

            self._init_finger_table(existing_node)

            # Propagate our arrival so existing nodes fix their finger tables.
            succ_transport = TSocket.TSocket(self.successor.ip, self.successor.port)
            succ_transport = TTransport.TBufferedTransport(succ_transport)
            succ_protocol = TBinaryProtocol.TBinaryProtocol(succ_transport)
            succ_client = compute.Client(succ_protocol)
            succ_transport.open()
            result = succ_client.fix_fingers(self_node)
            succ_transport.close()

            if result:
                print("Successfully joined the network")
            else:
                print("Failed to propagate join around the ring")
                sys.exit(1)

        except Exception as e:
            print(f"Error joining network: {e}")
            sys.exit(1)

    """find the successor node for a given ID."""
    def find_successor(self, id):
        curr_id = self.node_id
        current = node(self.ip, self.port, curr_id)
        
        # 1 node
        if self.successor.port == self.port:
            return current
        
        # current node is preceding id
        if self._is_between(id, curr_id, self.successor.id):
            return self.successor
        
        # forward request
        try:
            next_node = self.closest_preceding_node(id)
            if next_node is None:
                return self.successor
            
            transport = TSocket.TSocket(next_node.ip, next_node.port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            client = compute.Client(protocol)
            transport.open()
            result = client.find_successor(id)
            transport.close()
            
            return result
        
        except Exception as e:
            print(f"Error finding successor: {e}")
            return current  # Return self if there's an error

    """find the node preceding a given ID"""
    def find_predecessor(self, id):
        curr_id = self.node_id
        current = node(self.ip, self.port, curr_id)
        
        # one node in the network
        if self.successor.port == self.port:
            return current
        
        if self._is_between(id, curr_id, self.successor.id):
            return current
            
        try:
            next_node = self.closest_preceding_node(id)
            if next_node is None:
                return current
            # Connect to the next node
            transport = TSocket.TSocket(next_node.ip, next_node.port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            client = compute.Client(protocol)
            transport.open()
            result = client.find_predecessor(id)
            transport.close()
            
            return result
                
        except Exception as e:
            print(f"Error finding predecessor: {e}")
            return current
    
    
    """find the closest preceding node for a given hash."""
    def closest_preceding_node(self, hash):
        with self.lock:
            for i in range(FINGER_TABLE_SIZE - 1, -1, -1):
                finger_start = self.finger_table[i]["start"]
                finger_successor_id = self.finger_table[i]["successor_id"]
                
                if self._is_between(finger_successor_id, self.node_id, hash):
                    return self.finger_table[i]["node"]
        
        return None

    """return this node's predecessor"""
    def get_predecessor(self):
        with self.lock:
            return self.predecessor if self.predecessor else self.successor


    """helper to check if id is in the range (start, end]."""
    def _is_between(self, id, start, end):
        if start < end:
            return start < id <= end
        elif start > end:
            return id > start or id <= end
        else:
            return id != start

    """store and train a data file on the node responsible"""
    def put_data(self, filename):
        hash = hash_to_number(filename)
        print(f"Received put_data request for {filename}(hash: {hash}) at node: {self.node_id}")
        # base case
        if self._is_between(hash, self.predecessor.id, self.node_id):
            model = ML.mlp()
            if model.init_training_random(filename, NUM_CLASSES, HIDDEN_UNITS):
                model.set_momentum(MOMENTUM)
                print(f"file training at node {self.node_id}")
                t_err = model.train(LEARNING_RATE, TRAIN_EPOCHS)
            v, w = model.get_weights()
            curr_weights = weights(w, v, 0)
            with self.model_lock:
                self.models[filename] = curr_weights
        # forward to other node
        else:
            finger = self.closest_preceding_node(hash)
            # current node is closest preceding finger
            if finger is None:
                node = self.successor
            # forward to closest preceding finger  
            else:
                node = finger
            transport = TSocket.TSocket(node.ip, node.port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            client = compute.Client(protocol)
            transport.open()
            print(f"redirecting file to node {node.id}")
            client.put_data(filename)
            transport.close()
                
    """return a model weights for a given filename."""
    def get_model(self, filename):
        hash = hash_to_number(filename)
        print(f"Received get_model request for {filename} at node {self.node_id}, hash value: {hash}")
        # base case
        if self._is_between(hash, self.predecessor.id, self.node_id):
            print(f"model resides at node {self.node_id}")
            with self.model_lock:
                if filename in self.models.keys():
                    return self.models[filename]
                else:
                    # Return a weights object with wait status
                    return weights(w=[[0.0]], v=[[0.0]], status=1)
        # forward to other node
        else:
            finger = self.closest_preceding_node(hash)
            # current node is closest preceding finger
            if finger is None:
                node = self.successor
            # forward to closest preceding finger  
            else:
                node = finger
            transport = TSocket.TSocket(node.ip, node.port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            client = compute.Client(protocol)
            transport.open()
            print(f"redirecting request to node {node.id}")
            model = client.get_model(filename)
            transport.close()
            return model
        
    """Print information about this node"""
    def print_info(self):
        with self.lock:
            print("\n----- Node Information -----")
            print(f"Node ID: {self.node_id}")
            print(f"IP:Port: {self.ip}:{self.port}")
            
            if self.predecessor:
                print(f"Predecessor: {self.predecessor.ip}:{self.predecessor.port} (ID: {self.predecessor.id})")
            else:
                print("Predecessor: None")
                
            print(f"Successor: {self.successor.ip}:{self.successor.port} (ID: {self.successor.id})")
            
            print("\nFinger Table:")
            for i in range(FINGER_TABLE_SIZE):
                if self.finger_table[i]:
                    print(f" [{i}] - start: {self.finger_table[i]["start"]}, successor ID: {self.finger_table[i]["successor_id"]}, socket: {self.finger_table[i]["node"].ip}:{self.finger_table[i]["node"].port}")
                else:
                    print(f" [{i}] - None")
                    
            print("\nStored Data Files:")
            for file in self.models.keys():
                print(f"  - {file}")
                
            # print("\nStored Models:")
            # for filename in self.models:
            #     print(f"  - {filename}")
                
            print("----------------------------\n")
    
    """initialize the finget table of a new joined node"""
    def _init_finger_table(self, reference_node):
        with self.lock:
            try:
                transport = TSocket.TSocket(reference_node.ip, reference_node.port)
                transport = TTransport.TBufferedTransport(transport)
                protocol = TBinaryProtocol.TBinaryProtocol(transport)
                node_client = compute.Client(protocol)
                
                transport.open()
                
                self.finger_table[0]["start"] = (self.node_id + 1) % RING_SIZE
                successor = node_client.find_successor(self.node_id)
                self.finger_table[0]["successor_id"] = successor.id
                self.finger_table[0]["node"] = successor
                self.successor = successor
                
                succ_transport = TSocket.TSocket(self.successor.ip, self.successor.port)
                succ_transport = TTransport.TBufferedTransport(succ_transport)
                succ_protocol = TBinaryProtocol.TBinaryProtocol(succ_transport)
                succ_client = compute.Client(succ_protocol)
                
                succ_transport.open()
                
                # reassign successor's predecessor to be current node's predecessor
                pred = succ_client.get_predecessor()
                if pred.id != -1:
                    self.predecessor = pred
                
                # notify successor to change their predecessor
                succ_client.notify(node(self.ip, self.port, self.node_id))
                succ_client.print_info()
                succ_transport.close()
                
                # fill the rest of the finger table
                for i in range(1, FINGER_TABLE_SIZE):
                    self.finger_table[i]["start"] = (self.node_id + 2**i) % RING_SIZE
                    current_succ = node_client.find_successor((self.node_id + 2**i) % RING_SIZE)
                    if self.incorrect_entry_ft(self.node_id, self.finger_table[i]["start"], current_succ.id):
                        self.finger_table[i]["successor_id"] = self.node_id
                        self.finger_table[i]["node"] = node(self.ip, self.port, self.node_id)
                    else:
                        self.finger_table[i]["successor_id"] = current_succ.id
                        self.finger_table[i]["node"] = current_succ
                    
                transport.close()
                print("finger table initialized")
                self.print_info()
            except Exception as e:
                print(f"Error initializing finger table: {e}")
                raise
    
    """set a new predecessor if notified"""
    def notify(self, new_node):
        if not self.predecessor or self._is_between(new_node.id, self.predecessor.id, self.node_id):
            self.predecessor = new_node
        
    """recursively update finger tables in the ring"""
    def fix_fingers(self, new_node):
        new_id = new_node.id
        node = new_node
        with self.lock:
            try:
                for i in range(FINGER_TABLE_SIZE):
                    current = self.finger_table[i]
                    if self.incorrect_entry_ft(new_id, current["start"], current["successor_id"]):
                        current["successor_id"] = new_id
                        current["node"] = node
                        print(f"Updated finger[{i}]'s successor to {new_id}")
                        if i == 0:
                            self.successor = node

                self.print_info()
                # made a round
                if self.successor.id == new_id:
                    return True
                # call this on successor and continue the chain
                transport = TSocket.TSocket(self.successor.ip, self.successor.port)
                transport = TTransport.TBufferedTransport(transport)
                protocol = TBinaryProtocol.TBinaryProtocol(transport)
                succ_client = compute.Client(protocol)
                transport.open()
                result = succ_client.fix_fingers(new_node)
                transport.close()
                return result
                
            except Exception as e:
                print(f"Error fixing finger tables: {e}")
                
    """helper for fix_fingers to determine if the current entry is subject to be fixed"""
    def incorrect_entry_ft(self, id, start1, start2):
        if start1 < start2:
            return start1 <= id < start2
        elif start1 > start2:
            return id >= start1 or id < start2



def start_server(host, port, bootstrap_host=None, bootstrap_port=None):
    """Start the Thrift server for this compute node."""
    handler = ComputeHandler(host, port, bootstrap_host, bootstrap_port)
    processor = compute.Processor(handler)

    server_transport = TSocket.TServerSocket(port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TThreadedServer(processor, server_transport, tfactory, pfactory)

    print(f"Starting compute node server on port: {port}")
    server.serve()

if __name__ == "__main__":
    # First node:   python3 compute_server.py <host> <port>
    # Joining node: python3 compute_server.py <host> <port> <bootstrap_host> <bootstrap_port>
    if len(sys.argv) < 3:
        print("usage: python3 compute_server.py <host> <port> [bootstrap_host bootstrap_port]")
        sys.exit(1)
    host = sys.argv[1]
    port = int(sys.argv[2])
    bootstrap_host = sys.argv[3] if len(sys.argv) > 3 else None
    bootstrap_port = int(sys.argv[4]) if len(sys.argv) > 4 else None
    start_server(host, port, bootstrap_host, bootstrap_port)