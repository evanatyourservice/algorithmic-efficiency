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

TPU_VM_NAME=""
WANDB_API_KEY=""
WORKLOAD=""

# Display usage information
usage() {
  echo "Usage: $0 [-t|--tpu-vm TPU_VM_NAME] [-w|--workload WORKLOAD_NAME] [-k|--wandb-key WANDB_API_KEY]"
  echo "Available workloads: fastmri, imagenet_resnet, imagenet_vit, ogbg, criteo1tb, librispeech_conformer, librispeech_deepspeech, wmt"
  exit 1
}

# Parse command line arguments
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

# optional wandb key
if [ -n "$WANDB_API_KEY" ]; then
    WANDB_CMD="export WANDB_API_KEY=$WANDB_API_KEY"
    USE_WANDB_FLAG="--use_wandb"
else
    WANDB_CMD=""
    USE_WANDB_FLAG=""
fi

# workload
case $WORKLOAD in
  fastmri)
    COMMAND="python submission_runner.py --framework=jax --workload=fastmri --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/fastmri --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG 2>&1 | tee -a fastmri.log"
    ;;
  imagenet_resnet)
    COMMAND="python submission_runner.py --framework=jax --workload=imagenet_resnet --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/imagenet/jax --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --imagenet_v2_data_dir=/data/imagenet/jax --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG 2>&1 | tee -a imagenet_resnet.log"
    ;;
  imagenet_vit)
    COMMAND="python submission_runner.py --framework=jax --workload=imagenet_vit --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/imagenet/jax --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --imagenet_v2_data_dir=/data/imagenet/jax --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG 2>&1 | tee -a imagenet_vit.log"
    ;;
  ogbg)
    COMMAND="python submission_runner.py --framework=jax --workload=ogbg --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/ogbg --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG 2>&1 | tee -a ogbg.log"
    ;;
  criteo1tb)
    COMMAND="python submission_runner.py --framework=jax --workload=criteo1tb --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/criteo1tb --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG 2>&1 | tee -a criteo1tb.log"
    ;;
  librispeech_conformer)
    COMMAND="python submission_runner.py --framework=jax --workload=librispeech_conformer --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/librispeech --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --librispeech_tokenizer_vocab_path=/data/librispeech/spm_model.vocab --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG 2>&1 | tee -a conformer.log"
    ;;
  librispeech_deepspeech)
    COMMAND="python submission_runner.py --framework=jax --workload=librispeech_deepspeech --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/librispeech --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --librispeech_tokenizer_vocab_path=/data/librispeech/spm_model.vocab --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG 2>&1 | tee -a deepspeech.log"
    ;;
  wmt)
    COMMAND="python submission_runner.py --framework=jax --workload=wmt --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/wmt --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG 2>&1 | tee -a wmt.log"
    ;;
  *)
    echo "Error: Unknown workload '$WORKLOAD'"
    echo "Available workloads: fastmri, imagenet_resnet, imagenet_vit, ogbg, criteo1tb, librispeech_conformer, librispeech_deepspeech, wmt"
    exit 1
    ;;
esac

gcloud compute tpus tpu-vm ssh --zone "us-central2-b" "$TPU_VM_NAME" --project "mlcommons-algoperf" --worker=all --command "bash -c \"
$WANDB_CMD 
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
cd algorithmic-efficiency
nohup $COMMAND & 
PID=\\\$!
echo 'Background process started with PID '\\\$PID
disown \\\$PID
exit
\""