#!/usr/bin/env bash

docker build -t gcr.io/remy-demos/rapidsai_base:latest .

docker push gcr.io/remy-demos/rapidsai_base:latest