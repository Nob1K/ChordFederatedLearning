FROM python:3.12-slim

# the thrift compiler is needed to generate the rpc stubs during the build
RUN apt-get update \
    && apt-get install -y --no-install-recommends thrift-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# generate the rpc stubs
RUN thrift --gen py compute.thrift

ENV PYTHONUNBUFFERED=1
