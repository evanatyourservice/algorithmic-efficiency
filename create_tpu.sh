#!/bin/bash

# Example usage:
# ./create_tpu.sh
# ./create_tpu.sh --node-name=my-tpu
# ./create_tpu.sh --data-disk=my-data-disk
# ./create_tpu.sh --node-name=my-tpu --data-disk=my-data-disk

NODE_NAME="node-1"
DATA_DISK="dev-evan-1"

while [[ $# -gt 0 ]]; do
  case $1 in
    --node-name=*)
      NODE_NAME="${1#*=}"
      shift
      ;;
    --data-disk=*)
      DATA_DISK="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--node-name=NAME] [--data-disk=DISK_NAME]"
      exit 1
      ;;
  esac
done

gcloud compute tpus queued-resources create $NODE_NAME \
    --node-id $NODE_NAME \
    --project mlcommons-algoperf \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --runtime-version=tpu-ubuntu2204-base \
    --data-disk source=projects/mlcommons-algoperf/zones/us-central2-b/disks/$DATA_DISK,mode=read-only \
    --metadata-from-file='startup-script=_tpu_install.sh'

echo "TPU queued resource '$NODE_NAME' created with data disk '$DATA_DISK'"
