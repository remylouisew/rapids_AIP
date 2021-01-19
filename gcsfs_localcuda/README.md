## Submit AI Platform Training job using custom container

#bash build.sh

export REGION=us-central1
export JOB_NAME=rapids_job_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOB_NAME --region $REGION --config rapids_distributed.yaml

#gcloud ai-platform jobs stream-logs $JOB_NAME