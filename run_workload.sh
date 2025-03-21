#!/bin/bash

# Example usage:
# ./run_workload.sh -t node-1 -w fastmri
# ./run_workload.sh -t node-1 -w ogbg -k YOUR_WANDB_API_KEY
#
# Available workloads:
# - fastmri                  : FastMRI medical image reconstruction
# - imagenet_resnet          : ImageNet with ResNet architecture
# - imagenet_vit             : ImageNet with Vision Transformer
# - ogbg                     : Open Graph Benchmark
# - criteo1tb                : Criteo 1TB click prediction
# - librispeech_conformer    : LibriSpeech with Conformer
# - librispeech_deepspeech   : LibriSpeech with DeepSpeech
# - wmt                      : WMT machine translation

# NOTES
# algoperf was installed as root
# venv is at /algorithmic-efficiency/venv_py311
# logs saved to /algorithmic-efficiency

TPU_VM_NAME=""
WANDB_API_KEY=""
WORKLOAD=""

usage() {
  echo "Usage: $0 [-t|--tpu-vm TPU_VM_NAME] [-w|--workload WORKLOAD_NAME] [-k|--wandb-key WANDB_API_KEY]"
  echo "Available workloads: fastmri, imagenet_resnet, imagenet_vit, ogbg, criteo1tb, librispeech_conformer, librispeech_deepspeech, wmt"
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -t|--tpu-vm)
      TPU_VM_NAME="$2"
      shift 2
      ;;
    -k|--wandb-key)
      WANDB_API_KEY="$2"
      shift 2
      ;;
    -w|--workload)
      WORKLOAD="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

# required tpu vm name
if [ -z "$TPU_VM_NAME" ]; then
    echo "Error: TPU VM name is required"
    usage
fi

# required workload
if [ -z "$WORKLOAD" ]; then
    echo "Error: Workload name is required"
    usage
fi

WANDB_KEY_PARAM=""
if [ -n "$WANDB_API_KEY" ]; then
    WANDB_KEY_PARAM="$WANDB_API_KEY"
fi

echo "Starting workload $WORKLOAD on TPU VM $TPU_VM_NAME..."
# all users are given some safe sudo access on TPU VMs, so let's just let this run as root
gcloud compute tpus tpu-vm ssh --zone "us-central2-b" "$TPU_VM_NAME" --project "mlcommons-algoperf" --worker=all --command "sudo su && cd /algorithmic-efficiency && mkdir -p logs && nohup /algorithmic-efficiency/_run_workload.sh $WORKLOAD $WANDB_KEY_PARAM > logs/$WORKLOAD.log 2>&1 & echo \"Background process started with PID \$!\"; disown"
