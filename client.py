import sys
import os
import time

sys.path.append('gen-py')

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from compute import compute
from compute.ttypes import node, weights
from ML import ML
from config import NUM_CLASSES, HIDDEN_UNITS

# populate files
def get_files_in_directory(directory_path):
    file_paths = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))

    # sort for a deterministic shard set
    return sorted(file_paths)


# holds a connection to the ring and reconnects to another entry node if the
# current one dies, so a single node failure doesn't stop the client
class Ring:
    def __init__(self, entry_nodes):
        self.entry_nodes = entry_nodes
        self.client = None
        self.transport = None

    def connect(self):
        for host, port in self.entry_nodes:
            try:
                t = TSocket.TSocket(host, port)
                t.setTimeout(3000)
                t = TTransport.TBufferedTransport(t)
                c = compute.Client(TBinaryProtocol.TBinaryProtocol(t))
                t.open()
                self.client, self.transport = c, t
                print(f"connected to entry node {host}:{port}")
                return True
            except Thrift.TException:
                print(f"entry node {host}:{port} unreachable, trying next...")
        self.client = self.transport = None
        return False

    # run fn(client), reconnecting to another entry node if the call fails
    def call(self, fn, attempts=15):
        for _ in range(attempts):
            if self.client is None and not self.connect():
                time.sleep(2)
                continue
            try:
                return fn(self.client)
            except Thrift.TException:
                print("entry node lost, reconnecting...")
                try:
                    self.transport.close()
                except Exception:
                    pass
                self.client = None
                time.sleep(2)
        raise RuntimeError("no entry node reachable")

    def close(self):
        if self.transport is not None:
            try:
                self.transport.close()
            except Exception:
                pass


def main():

    if len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print("usage: python3 client.py <node_host> <node_port> [<node_host2> <node_port2> ...]")
        return
    # entry nodes are tried in order until one answers
    entry_nodes = [(sys.argv[i], int(sys.argv[i + 1])) for i in range(1, len(sys.argv), 2)]
    training_files = get_files_in_directory("letters")[:-15]

    ring = Ring(entry_nodes)
    if not ring.connect():
        print("No entry node reachable")
        return

    # init the shared model
    mlp = ML.mlp()
    mlp.init_training_random(training_files[0], NUM_CLASSES, HIDDEN_UNITS)
    shared_v, shared_w = mlp.get_weights()
    shared_v = ML.scale_matricies(shared_v, 0)
    shared_w = ML.scale_matricies(shared_w, 0)

    # send out all training files
    for file in training_files:
        ring.call(lambda c: c.put_data(file))
    # head start for training; anything not ready is caught by the poll below
    print("Done providing data, sleeping to let the models train")
    time.sleep(30)
    # pull each model and aggregate it. status 0 = ready, 1 = still training,
    # -1 = the ring is mid-repair, so keep polling until it's ready
    for file in training_files:
        model = ring.call(lambda c: c.get_model(file))
        while model.status != 0:
            print(f"waiting for {file}'s model (status {model.status})")
            time.sleep(5)
            model = ring.call(lambda c: c.get_model(file))
        shared_v = ML.sum_matricies(shared_v, model.v)
        shared_w = ML.sum_matricies(shared_w, model.w)
    # validate
    shared_v = ML.scale_matricies(shared_v, 1/len(training_files))
    shared_w = ML.scale_matricies(shared_w, 1/len(training_files))
    mlp.set_weights(shared_v, shared_w)
    v_err = mlp.validate("validate_letters.txt")
    print("final validation error:", v_err)
    ring.close()


if __name__ == '__main__':
    try:
        main()
    except Thrift.TException as tx:
        print('%s' % tx.message)
