#!/bin/bash

# Example usage:
# ./create_tpu.sh
# ./create_tpu.sh -n node-1 -d dev-evan-1

NODE_NAME="node-1"
DATA_DISK="dev-evan-1"

usage() {
  echo "Usage: $0 [-n|--node-name NODE_NAME] [-d|--data-disk DISK_NAME]"
  exit 1
}

# Parse command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    -n|--node-name)
      NODE_NAME="$2"
      shift 2
      ;;
    -d|--data-disk)
      DATA_DISK="$2"
      shift 2
      ;;
    *)
      usage
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
