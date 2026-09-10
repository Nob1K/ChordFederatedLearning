"""Shared config for the chord ring and the compute cluster.

Imported by every compute node and the client so the id space and the MLP
hyperparameters are defined in one place.
"""

# bits in the chord id space. nodes hash "host:port" into [0, RING_SIZE), so a
# big space (2**16 slots) keeps id collisions unlikely
M = 16
RING_SIZE = 2 ** M

# Number of finger-table entries maintained by each node
FINGER_TABLE_SIZE = M

# Number of successors each node tracks (not just the immediate one). If a
# node's successor dies, it fails over to the next entry in this list. Also for replication: a trained model is copied to
# this many successors.
SUCCESSOR_LIST_SIZE = 3

# ring-maintenance and RPC timeout config (sec)
STABILIZE_INTERVAL = 1.0
RPC_TIMEOUT = 2.0

# MLP hyperparameters. shared by the nodes (local training) and the client
# (the model it aggregates into). NUM_CLASSES and HIDDEN_UNITS must match
# across all of them or the weight matrices won't line up for FedAvg.
NUM_CLASSES = 26
HIDDEN_UNITS = 100    # hidden-layer width
LEARNING_RATE = 0.03
TRAIN_EPOCHS = 1000
MOMENTUM = 0.7
