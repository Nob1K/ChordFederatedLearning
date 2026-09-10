#!/usr/bin/env bash
#
# Starts the ring, runs the client, and kills one node while it is still
# training. The node's shards get rebuilt (from a backup copy, or by retraining
# from the file) so the client still finishes at around 0.29.
#
#   ./demo/docker-chaos.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

NODE=compute2   # the node we kill part way through

cleanup() { docker compose down -v >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> building and starting the three nodes"
docker compose up -d --build compute0 compute1 compute2

echo "==> waiting for the ring to settle"
sleep 12
echo "--- ring state per node ---"
for n in compute0 compute1 compute2; do
  echo -n "$n: "; docker compose logs "$n" 2>&1 | grep '\[ring\]' | tail -1 || echo "(no state yet)"
done

# a few seconds after the client starts, kill one node while training is running
( sleep 8 && echo && echo ">>> killing $NODE while it is training <<<" && docker compose kill "$NODE" ) &

echo
echo "==> running the client ($NODE gets killed part way through)"
docker compose run --rm --no-deps client \
  python client.py compute0 9000 compute1 9001 compute2 9002

echo
echo "--- ring state after the kill ($NODE should be gone) ---"
for n in compute0 compute1; do
  echo -n "$n: "; docker compose logs "$n" 2>&1 | grep '\[ring\]' | tail -1 || echo "(none)"
done
echo
echo "==> a 'final validation error: ~0.29' above means the recovery worked"
