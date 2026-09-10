.PHONY: gen install test clean docker-up docker-down demo demo-chaos

# Regenerate the Thrift RPC stubs into gen-py/ (requires the `thrift` compiler)
gen:
	thrift --gen py compute.thrift

# Create a local virtualenv, install deps, and generate stubs
install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	$(MAKE) gen

# Run the end-to-end tests (ring forms, heals on crash, survives mid-run crash)
test:
	.venv/bin/python tests/churn_test.py

# Bring the whole cluster up / down with docker compose
docker-up:
	docker compose up --build

docker-down:
	docker compose down -v

# Run the cluster in docker (3 nodes + client)
demo:
	docker compose up --build

# Same run, but kill a node mid-training to show it recovers
demo-chaos:
	bash demo/docker-chaos.sh

clean:
	rm -rf gen-py __pycache__ */__pycache__ *.log tests/logs
