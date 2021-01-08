#!/usr/bin/env bash

docker build -t gcr.io/k80-exploration/gcp_rapids_dist:fuse .

docker push gcr.io/k80-exploration/gcp_rapids_dist:fuse
