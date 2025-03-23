#!/bin/bash

# this file is run on the TPU VM

WORKLOAD=$1
WANDB_API_KEY=$2

if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY=$WANDB_API_KEY
    USE_WANDB_FLAG="--use_wandb"
else
    USE_WANDB_FLAG=""
fi

cd /algorithmic-efficiency
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
source /algorithmic-efficiency/venv_py311/bin/activate
mkdir -p experiment_runs

# step hints:
# fastmri 18_094
# imagenet_resnet 195_999
# imagenet_vit 167_999
# ogbg 52_000
# criteo1tb 10_666
# librispeech_conformer 76_000
# librispeech_deepspeech 38_400
# wmt 120_000

case $WORKLOAD in
  fastmri)
    COMMAND="python submission_runner.py --framework=jax --workload=fastmri --max_global_steps=18094 --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/fastmri --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    ;;
  imagenet_resnet)
    COMMAND="python submission_runner.py --framework=jax --workload=imagenet_resnet --max_global_steps=195999 --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/imagenet/jax --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --imagenet_v2_data_dir=/data/imagenet/jax --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    ;;
  imagenet_vit)
    COMMAND="python submission_runner.py --framework=jax --workload=imagenet_vit --max_global_steps=167999 --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/imagenet/jax --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --imagenet_v2_data_dir=/data/imagenet/jax --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    ;;
  ogbg)
    COMMAND="python submission_runner.py --framework=jax --workload=ogbg --max_global_steps=52000 --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/ogbg --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    ;;
  criteo1tb)
    COMMAND="python submission_runner.py --framework=jax --workload=criteo1tb --max_global_steps=10666 --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/criteo1tb --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    ;;
  librispeech_conformer)
    COMMAND="python submission_runner.py --framework=jax --workload=librispeech_conformer --max_global_steps=76000 --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/librispeech --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --librispeech_tokenizer_vocab_path=/data/librispeech/spm_model.vocab --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    ;;
  librispeech_deepspeech)
    COMMAND="python submission_runner.py --framework=jax --workload=librispeech_deepspeech --max_global_steps=38400 --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/librispeech --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --librispeech_tokenizer_vocab_path=/data/librispeech/spm_model.vocab --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
    ;;
  wmt)
    COMMAND="python submission_runner.py --framework=jax --workload=wmt --max_global_steps=120000 --submission_path=prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py --data_dir=/mnt/disks/persist/algoperf_data/wmt --num_tuning_trials=1 --experiment_dir=experiment_runs --experiment_name=tests/regression_tests/adamw --overwrite=True --save_checkpoints=False --tuning_ruleset=external --tuning_search_space=prize_qualification_baselines/external_tuning/tuning_search_space.json $USE_WANDB_FLAG"
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