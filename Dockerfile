FROM python:3.12-slim

# Thrift compiler, used to generate the Python RPC stubs at build time
RUN apt-get update \
    && apt-get install -y --no-install-recommends thrift-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# generate gen-py inside the image
RUN thrift --gen py compute.thrift

ENV PYTHONUNBUFFERED=1
