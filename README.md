# Chord Federated Learning

A distributed **federated-learning** system built on top of a **Chord distributed hash table (DHT)**.
Training data is consistent-hashed and routed across a ring of compute nodes; each node trains a
local neural network on the shards it is responsible for, and a client aggregates all the local
models into a single global model using **federated averaging (FedAvg)**.

The project combines three areas:

- **Distributed systems**: a coordinator-free Chord ring with finger-table routing (`O(log N)` lookups), self-assigned node IDs, peer bootstrap, and successor/predecessor maintenance.
- **RPC / serialization**: all node-to-node and client-to-node communication uses [Apache Thrift](https://thrift.apache.org/).
- **Machine learning**: a multi-layer perceptron (forward/backprop, softmax, momentum) implemented from scratch in NumPy, trained on a 26-class letter-recognition dataset.

## Architecture

```mermaid
flowchart TB
    subgraph client_side [ ]
        C[Client<br/>pushes shards · FedAvg · validation]
    end

    subgraph ring [Chord ring · identifier space 0..65535]
        N0((node 0x2481))
        N1((node 0xa678))
        N2((node 0xfba5))
        N0 -->|successor| N1 -->|successor| N2 -->|successor| N0
    end

    C -->|put_data / get_model<br/>via any entry node| N1
    N1 -.->|consistent-hash routing<br/>via finger tables| N0
    N1 -.-> N2
```

There is **no central coordinator**. Each node assigns its own Chord ID by
hashing its `host:port`, and a joining node bootstraps through any existing peer
given on the command line (the first node forms the ring alone).

**Roles**

| Component | File | Responsibility |
|-----------|------|----------------|
| Compute node | `compute_server.py` | Self-assigns a Chord ID, joins via a bootstrap peer (or forms the ring), maintains a finger table + successor/predecessor, routes `put_data`/`get_model` by hash, and trains a local MLP on the shards it owns. |
| Client | `client.py` | Connects to any entry node, pushes all training shards into the ring, waits for local training, pulls each trained model back, aggregates with FedAvg, and reports validation error. |
| ML library | `ML/ML.py` | From-scratch NumPy MLP (ReLU hidden layer, softmax output, momentum SGD). |
| Shared config | `config.py` | Chord identifier space (`M`, `RING_SIZE`) and ML hyperparameters. |

## How it works

1. **Naming.** Each compute node assigns itself an ID by hashing `host:port` into the identifier
   space `[0, RING_SIZE)` — no allocator. The large space (`2**16`) makes collisions negligible.
2. **Join.** The first node forms the ring alone. A later node is told one existing peer on the
   command line, initializes its finger table through it, fixes its successor/predecessor, and
   propagates the new entry around the ring (`fix_fingers`).
3. **Data placement.** A file name is hashed (SHA-1 → `mod RING_SIZE`). `put_data` routes the file to
   the node responsible for that key using the finger table, and that node trains an MLP on it.
4. **Aggregation.** The client calls `get_model` for every shard (polling until training completes),
   sums the weight matrices, and scales by `1/N` to produce the FedAvg global model, then validates it.

## Quickstart

### Option A — Docker (recommended)

Brings up 3 compute nodes + client on an isolated network. No local Python or Thrift
compiler needed.

```bash
docker compose up --build
```

You'll see the ring form and, after training, a line like `final validation error: 0.29`
(~71% validation accuracy on the 26-class letter-recognition task).
Tear down with `docker compose down -v`.

### Option B — Local (virtualenv)

Requires Python 3.12+ and the Thrift compiler (`thrift`) on your `PATH`.

```bash
make install          # create .venv, install deps, generate gen-py/
source .venv/bin/activate

# in separate terminals (or backgrounded), from the repo root:
# first node forms the ring; the rest bootstrap through it (host port [bootstrap_host bootstrap_port])
python compute_server.py 127.0.0.1 9000
python compute_server.py 127.0.0.1 9001 127.0.0.1 9000
python compute_server.py 127.0.0.1 9002 127.0.0.1 9000
# the client takes one or more entry nodes and tries them in order
python client.py 127.0.0.1 9000 127.0.0.1 9001 127.0.0.1 9002
```

## Project layout

```
compute.thrift                      Thrift service + struct definitions (source of truth for the RPC API)
config.py                           Chord identifier space (M, RING_SIZE) and ML hyperparameters
compute_server.py                   Chord node: self-naming, join, routing, ring maintenance, local training
client.py                           Driver: shard distribution, FedAvg aggregation, validation
ML/ML.py                            From-scratch NumPy MLP
letters/ , validate_letters.txt     Letter-recognition training shards + validation set
gen-py/                             Thrift-generated stubs (gitignored; run `make gen`)
Dockerfile / docker-compose.yml     Containerized multi-node cluster
docs/churn-plan.md                  Roadmap: removing the coordinator + churn tolerance
```

## Design notes

- **Identifier space.** Node IDs and key hashes share a single space `[0, RING_SIZE)` with
  `RING_SIZE = 2**M` (`config.py`). The power-of-two size is what makes finger offsets `2**i` tile the
  ring correctly.
- **Self-assigned IDs.** Nodes name themselves by hashing `host:port`, so there is no coordinator to
  hand out IDs or entry points. A joiner only needs the address of one existing peer.
- **Training.** The MLP uses a batch-mean gradient (so the learning rate is independent of shard size)
  and vectorized forward/backprop. Hyperparameters live in `config.py` and were tuned with an offline
  FedAvg harness; the shipped config reaches ~71% validation accuracy. Because the shards are IID and
  every local model starts from the same seed, FedAvg weight-averaging behaves like a mild ensemble and
  slightly *beats* the average individual model.
