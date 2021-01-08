#!/usr/bin/env bash

docker build -t gcr.io/remy-demos/rapids_gcsfs:latest .

docker push gcr.io/remy-demos/rapids_gcsfs:latest
