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

# open a client to the first reachable entry node. Any node in the ring
# is a valid entry point; put_data/get_model are routed internally by hash.
def connect_entry_node(entry_nodes):
    for host, port in entry_nodes:
        try:
            transport = TSocket.TSocket(host, port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            client = compute.Client(protocol)
            transport.open()
            print(f"Connected to entry node {host}:{port}")
            return client, transport
        except Thrift.TException:
            print(f"Entry node {host}:{port} unreachable, trying next...")
    return None, None

def main():

    if len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print("usage: python3 client.py <node_host> <node_port> [<node_host2> <node_port2> ...]")
        return
    # one or more entry nodes; the client tries them in order until one answers
    entry_nodes = [(sys.argv[i], int(sys.argv[i + 1])) for i in range(1, len(sys.argv), 2)]
    training_files = get_files_in_directory("letters")[:-15]

    node_client, node_transport = connect_entry_node(entry_nodes)
    if node_client is None:
        print("❌ No entry node reachable")
        return

    # initialize shared ML model
    mlp = ML.mlp()
    mlp.init_training_random(training_files[0], NUM_CLASSES, HIDDEN_UNITS)
    shared_v, shared_w = mlp.get_weights()
    shared_v = ML.scale_matricies(shared_v, 0)
    shared_w = ML.scale_matricies(shared_w, 0)

    # send out all training files
    for file in training_files:
        node_client.put_data(file)
    # give the nodes a head start on training; any models not ready yet are
    # picked up by the status==1 polling loop below
    print("Done providing data, sleeping to let the models train")
    time.sleep(30)
    # acquire and aggregate each model
    for file in training_files:
        model = node_client.get_model(file)
        while model.status == 1:
            print(f"waiting for file {file}'s model to be done")
            time.sleep(10)
            model = node_client.get_model(file)
        shared_v = ML.sum_matricies(shared_v, model.v)
        shared_w = ML.sum_matricies(shared_w, model.w)
    # validate
    shared_v = ML.scale_matricies(shared_v, 1/len(training_files))
    shared_w = ML.scale_matricies(shared_w, 1/len(training_files))
    mlp.set_weights(shared_v, shared_w)
    v_err = mlp.validate("validate_letters.txt")
    print("final validation error:", v_err)
    node_transport.close()
    



if __name__ == '__main__':
    try:
        main()
    except Thrift.TException as tx:
        print('%s' % tx.message)