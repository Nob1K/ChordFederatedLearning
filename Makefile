.PHONY: gen install clean docker-up docker-down

# Regenerate the Thrift RPC stubs into gen-py/ (requires the `thrift` compiler)
gen:
	thrift --gen py compute.thrift

# Create a local virtualenv, install deps, and generate stubs
install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	$(MAKE) gen

# Bring the whole cluster up / down with docker compose
docker-up:
	docker compose up --build

docker-down:
	docker compose down -v

clean:
	rm -rf gen-py __pycache__ */__pycache__ *.log
