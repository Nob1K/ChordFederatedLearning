"""Shared configuration for the Chord ring and the compute cluster.

Both the supernode and the compute nodes import these constants so the
identifier space is defined in exactly one place.
"""

# Number of bits in the Chord identifier space
M = 4
RING_SIZE = 2 ** M

# Number of finger-table entries maintained by each node
FINGER_TABLE_SIZE = M

# Maximum number of compute nodes the supernode will admit into the ring.
# This is a capacity limit on the cluster, distinct from the ring size above,
# and must satisfy MAX_NODES <= RING_SIZE
MAX_NODES = 10

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
