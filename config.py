"""Shared configuration for the Chord ring and the compute cluster.

Every compute node and the client import these constants so the identifier
space is defined in exactly one place.
"""

# Number of bits in the Chord ID space
#
# Nodes assign their own IDs by hashing "host:port" into [0, RING_SIZE). A large
# space (2**16 = 65,536 slots) makes it unlikely that two addresses land on the same slot
M = 16
RING_SIZE = 2 ** M

# Number of finger-table entries maintained by each node
FINGER_TABLE_SIZE = M

# Number of successors each node tracks (not just the immediate one). If a
# node's successor dies, it fails over to the next entry in this list. Also for replication: a trained model is copied to
# this many successors. (useful after refactoring churn tolerance)
SUCCESSOR_LIST_SIZE = 3

# ring-maintenance and RPC timeout config (sec). (useful after refactoring churn tolerance)
STABILIZE_INTERVAL = 1.0
RPC_TIMEOUT = 2.0

# ---------------------------------------------------------------------------
# MLP hyperparameters
#
# These are shared by every compute node (which trains local models) and the
# client (which allocates the shared model it aggregates into). NUM_CLASSES and
# HIDDEN_UNITS in particular MUST match across all of them, otherwise the weight
# matrices won't have compatible shapes for FedAvg.
# ---------------------------------------------------------------------------
NUM_CLASSES = 26 
HIDDEN_UNITS = 100    # hidden-layer width
LEARNING_RATE = 0.03
TRAIN_EPOCHS = 1000
MOMENTUM = 0.7
