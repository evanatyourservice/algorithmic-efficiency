#!/bin/bash

WORKLOAD=$1
WANDB_API_KEY=$2

cd /algorithmic-efficiency
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
source /algorithmic-efficiency/venv_py311/bin/activate

if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY=$WANDB_API_KEY
    USE_WANDB_FLAG="--use_wandb"
else
    USE_WANDB_FLAG=""
fi

case $WORKLOAD in
  fastmri)
    COMMAND="python submission_runner.py --framework=jax --workload=fastmri --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/fastmri --num_tuning_trials=1 --experiment_dir=~/experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    LOG_FILE="fastmri.log"
    ;;
  imagenet_resnet)
    COMMAND="python submission_runner.py --framework=jax --workload=imagenet_resnet --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/imagenet/jax --num_tuning_trials=1 --experiment_dir=~/experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --imagenet_v2_data_dir=/data/imagenet/jax --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    LOG_FILE="imagenet_resnet.log"
    ;;
  imagenet_vit)
    COMMAND="python submission_runner.py --framework=jax --workload=imagenet_vit --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/imagenet/jax --num_tuning_trials=1 --experiment_dir=~/experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --imagenet_v2_data_dir=/data/imagenet/jax --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    LOG_FILE="imagenet_vit.log"
    ;;
  ogbg)
    COMMAND="python submission_runner.py --framework=jax --workload=ogbg --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/ogbg --num_tuning_trials=1 --experiment_dir=~/experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    LOG_FILE="ogbg.log"
    ;;
  criteo1tb)
    COMMAND="python submission_runner.py --framework=jax --workload=criteo1tb --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/criteo1tb --num_tuning_trials=1 --experiment_dir=~/experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    LOG_FILE="criteo1tb.log"
    ;;
  librispeech_conformer)
    COMMAND="python submission_runner.py --framework=jax --workload=librispeech_conformer --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/librispeech --num_tuning_trials=1 --experiment_dir=~/experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --librispeech_tokenizer_vocab_path=/data/librispeech/spm_model.vocab --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    LOG_FILE="conformer.log"
    ;;
  librispeech_deepspeech)
    COMMAND="python submission_runner.py --framework=jax --workload=librispeech_deepspeech --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/librispeech --num_tuning_trials=1 --experiment_dir=~/experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --librispeech_tokenizer_vocab_path=/data/librispeech/spm_model.vocab --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    LOG_FILE="deepspeech.log"
    ;;
  wmt)
    COMMAND="python submission_runner.py --framework=jax --workload=wmt --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/wmt --num_tuning_trials=1 --experiment_dir=~/experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    LOG_FILE="wmt.log"
    ;;
  *)
    echo "Error: Unknown workload '$WORKLOAD'"
    echo "Available workloads: fastmri, imagenet_resnet, imagenet_vit, ogbg, criteo1tb, librispeech_conformer, librispeech_deepspeech, wmt"
    exit 1
    ;;
esac

echo "Running workload: $WORKLOAD"
echo "Command: $COMMAND"
eval $COMMAND