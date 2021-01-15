#!/usr/bin/env bash

docker build -t gcr.io/gpu-test-project/rapids_gcsfs:latest .

docker push gcr.io/gpu-test-project/rapids_gcsfs:latest
