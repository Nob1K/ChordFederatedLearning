#!/usr/bin/env python3
"""End-to-end tests for the chord federated-learning ring.

Starts compute nodes and a client as local processes and checks:
  1. the ring forms and a full training run finishes with good accuracy
  2. the ring heals after a node crash and lookups still resolve
  3. a training run still finishes when a node is killed mid-training

Run from the repo root with the project venv:
    .venv/bin/python tests/churn_test.py
Node/client logs are written to tests/logs/.
"""
import os
import re
import sys
import time
import socket
import hashlib
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gen-py"))
LOGDIR = os.path.join(ROOT, "tests", "logs")
os.makedirs(LOGDIR, exist_ok=True)

from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from compute import compute

RING_SIZE = 2 ** 16
HOST = "127.0.0.1"
PORTS = [9200, 9201, 9202]


def key_hash(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % RING_SIZE


def log(name):
    return os.path.join(LOGDIR, name)


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex((HOST, port)) != 0


def wait_port(port, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((HOST, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def rpc(port, fn, timeout=2.0):
    t = TSocket.TSocket(HOST, port)
    t.setTimeout(int(timeout * 1000))
    t = TTransport.TBufferedTransport(t)
    c = compute.Client(TBinaryProtocol.TBinaryProtocol(t))
    t.open()
    try:
        return fn(c)
    finally:
        t.close()


def find_owner(port, h):
    return rpc(port, lambda c: c.find_successor(h))


def successor_port(port):
    sl = rpc(port, lambda c: c.get_successor_list())
    return sl[0].port if sl else None


def ring_is_cycle(ports):
    # follow successor pointers and confirm they form one loop over all ports
    try:
        succ = {p: successor_port(p) for p in ports}
    except Exception:
        return False
    if any(succ[p] not in ports for p in ports):
        return False
    cur, visited = ports[0], []
    for _ in range(len(ports)):
        visited.append(cur)
        cur = succ[cur]
    return cur == ports[0] and sorted(visited) == sorted(ports)


def wait_ring(ports, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if ring_is_cycle(ports):
            return True
        time.sleep(1)
    return False


def start_node(port, bootstrap=None):
    args = [sys.executable, "compute_server.py", HOST, str(port)]
    if bootstrap:
        args += [HOST, str(bootstrap)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    f = open(log(f"node_{port}.log"), "w")
    return subprocess.Popen(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)


def start_cluster():
    procs = {PORTS[0]: start_node(PORTS[0])}
    wait_port(PORTS[0])
    for p in PORTS[1:]:
        procs[p] = start_node(p, PORTS[0])
        wait_port(p)
    return procs


def start_client(entry_ports, logname):
    args = [sys.executable, "client.py"]
    for p in entry_ports:
        args += [HOST, str(p)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    path = log(logname)
    f = open(path, "w")
    return subprocess.Popen(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env), path


def parse_error(path):
    m = re.search(r"final validation error:\s*([0-9.]+)", open(path).read())
    return float(m.group(1)) if m else None


def kill(proc):
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


MAX_ERR = 0.32  # deterministic run is 0.2914; this leaves a safe margin


def test_normal():
    """ring forms and a full training run reaches good accuracy"""
    procs = start_cluster()
    try:
        assert wait_ring(PORTS, 30), "ring did not converge into one cycle"
        client, path = start_client(PORTS, "client_normal.log")
        client.wait(timeout=200)
        err = parse_error(path)
        assert err is not None, "client never reported a validation error"
        assert err < MAX_ERR, f"validation error too high: {err}"
        return f"ring converged; client finished at error {err:.4f}"
    finally:
        for p in procs.values():
            kill(p)


def test_heal_on_crash():
    """after a node is killed, the survivors re-form and lookups stay live"""
    procs = start_cluster()
    try:
        assert wait_ring(PORTS, 30), "ring did not converge"
        dead = PORTS[2]
        kill(procs.pop(dead))
        survivors = [PORTS[0], PORTS[1]]
        assert wait_ring(survivors, 30), "survivors did not re-form a ring"
        bad = 0
        for i in range(30):
            if find_owner(PORTS[0], key_hash(f"probe-{i}")).port == dead:
                bad += 1
        assert bad == 0, f"{bad}/30 lookups resolved to the dead node"
        return "survivors re-formed a 2-node ring; 30/30 lookups resolved to live nodes"
    finally:
        for p in procs.values():
            kill(p)


def test_training_survives_crash():
    """a node killed mid-training does not stop the client from finishing"""
    procs = start_cluster()
    client = None
    try:
        assert wait_ring(PORTS, 30), "ring did not converge"
        client, path = start_client(PORTS, "client_churn.log")
        time.sleep(7)  # let the client push data and training get going
        kill(procs.pop(PORTS[2]))  # kill a data node mid-training
        client.wait(timeout=220)
        err = parse_error(path)
        assert err is not None, "client did not finish after the crash"
        assert err < MAX_ERR, f"validation error too high after crash: {err}"
        return f"client finished despite a mid-training crash; error {err:.4f}"
    finally:
        if client is not None:
            kill(client)
        for p in procs.values():
            kill(p)


TESTS = [
    ("normal ring + full training run", test_normal),
    ("ring heals after a crash", test_heal_on_crash),
    ("training survives a mid-run crash", test_training_survives_crash),
]


def main():
    busy = [p for p in PORTS if not port_free(p)]
    if busy:
        print(f"ports already in use: {busy} (is a cluster still running?)")
        return 1

    print(f"running {len(TESTS)} tests, logs in tests/logs/\n")
    results = []
    for name, fn in TESTS:
        print(f"... {name}")
        t0 = time.time()
        try:
            detail = fn()
            results.append((name, True, detail, time.time() - t0))
            print(f"    PASS ({time.time() - t0:.0f}s): {detail}\n")
        except Exception as e:
            results.append((name, False, str(e), time.time() - t0))
            print(f"    FAIL ({time.time() - t0:.0f}s): {e}\n")
        time.sleep(2)  # let ports free up between tests

    passed = sum(1 for _, ok, _, _ in results if ok)
    print("=" * 60)
    for name, ok, detail, dt in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt:.0f}s)")
    print(f"{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
