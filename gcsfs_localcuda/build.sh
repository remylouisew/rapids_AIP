#!/usr/bin/env bash

docker build -t gcr.io/remy-sandbox/rapids_gcsfs:latest .

docker push gcr.io/remy-sandbox/rapids_gcsfs:latest
