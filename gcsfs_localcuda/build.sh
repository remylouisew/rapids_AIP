#!/usr/bin/env bash
# Build the Rapids-based custom container with the XGboost training code
## Change the gcr.io path to your own repository!

docker build -t gcr.io/remy-sandbox/rapids_gcsfs:latest .

docker push gcr.io/remy-sandbox/rapids_gcsfs:latest

#docker build -t gcr.io/remy-sandbox/rapids_gcsfs:noarg .
#docker push gcr.io/remy-sandbox/rapids_gcsfs:noarg