from typing import Optional
import os
import time
from dataclasses import dataclass
from types import MappingProxyType
from inspect import signature
from datetime import datetime
from absl import app, flags, logging
import optuna
from optuna.integration import WeightsAndBiasesCallback

from algoperf import spec
from algoperf import random_utils as prng
from algoperf.workloads import workloads

from submissions.external_tuning.kron import submission


FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'framework',
    'jax',
    enum_values=['jax', 'pytorch'],
    help='Whether to use Jax or Pytorch for the submission.')
flags.DEFINE_string(
    'workload',
    'ogbg',
    'Name of the workload to tune')
flags.DEFINE_string(
    'data_dir',
    '/dev/shm/ogbg',
    'Path to the data directory')
flags.DEFINE_string(
    'experiment_dir',
    f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    'Experiment output directory')
flags.DEFINE_float(
    'step_hint_factor',
    0.05,
    'Fraction of step hint to use for tuning')
flags.DEFINE_integer(
    'num_tuning_trials',
    10,
    'Number of tuning trials')
flags.DEFINE_string(
    'imagenet_v2_data_dir',
    None,
    'Dataset location for ImageNet-v2.')
flags.DEFINE_integer(
    'rng_seed',
    42,
    'Value of rng seed. If None, a random seed will be generated.')
flags.DEFINE_boolean(
    'use_wandb',
    True,
    'Whether to use Weights & Biases logging.')

@dataclass
class Hyperparameters:
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    preconditioner_lr: float = 0.5
    label_smoothing: float = 0.0
    dropout_rate: float = 0.0
    b1: float = 0.9
    flat_start: int = 1000
    min_prob: float = 0.05
    preconditioner_init_scale: float = 1.0
    warmup_factor: float = 0.0
    max_size_triangular: int = 8192
    min_ndim_triangular: int = 2
    memory_save_mode: Optional[bool] = "all_diag"
    precond_update_precision: str = "tensorfloat32"
    precond_grads_precision: Optional[str] = None
    lax_map_scanned_layers: bool = False
    lax_map_batch_size: int = 8
    merge_small_dims: bool = True
    target_merged_dim_size: int = 4096
    partition_grads_into_blocks: bool = True
    block_size: int = 256


def get_tuned_hyperparameters(trial: optuna.Trial) -> Hyperparameters:
    return Hyperparameters(
        learning_rate=trial.suggest_float('learning_rate', 0.0002, 0.002, log=True),
        weight_decay=trial.suggest_float('weight_decay', 0.0001, 1.0, log=True),
        preconditioner_lr=trial.suggest_float('preconditioner_lr', 0.3, 1.5),
        label_smoothing=trial.suggest_float('label_smoothing', 0.0, 0.25),
        dropout_rate=trial.suggest_float('dropout_rate', 0.0, 0.5),
    )


workload_info = {
    "criteo1tb": {
        "metric_name": "loss",
        "maximize": False,
        "step_hint": 10_666,
        "target_metric_name": "loss",
    },
    "fastmri": {
        "metric_name": "ssim",
        "maximize": True,
        "step_hint": 18_094,
        "target_metric_name": "ssim",
    },
    "imagenet_resnet": {
        "metric_name": "accuracy",
        "maximize": True,
        "step_hint": 195_999,
        "target_metric_name": "accuracy",
    },
    "imagenet_vit": {
        "metric_name": "accuracy",
        "maximize": True,
        "step_hint": 167_999,
        "target_metric_name": "accuracy",
    },
    "librispeech_conformer": {
        "metric_name": "wer",
        "maximize": False,
        "step_hint": 76_000,
        "target_metric_name": "wer",
    },
    "librispeech_deepspeech": {
        "metric_name": "wer",
        "maximize": False,
        "step_hint": 38_400,
        "target_metric_name": "wer",
    },
    "ogbg": {
        "metric_name": "mean_average_precision",
        "maximize": True,
        "step_hint": 52_000,
        "target_metric_name": "mean_average_precision",
    },
    "wmt": {
        "metric_name": "bleu",
        "maximize": True,
        "step_hint": 120_000,
        "target_metric_name": "bleu",
    },
    "mnist": {
        "metric_name": "accuracy",
        "maximize": True,
        "step_hint": 7813,
        "target_metric_name": "accuracy",
    },
    "cifar": {
        "metric_name": "accuracy",
        "maximize": True,
        "step_hint": 4883,
        "target_metric_name": "accuracy",
    },
}


def get_metric_value(result, metric_name):
    # First try direct access
    if metric_name in result:
        return result[metric_name]
    # Fallback to checking validation split
    if f'validation/{metric_name}' in result:
        return result[f'validation/{metric_name}']
    # If still not found, show available keys
    available_keys = ', '.join(result.keys())
    raise KeyError(
        f"Metric '{metric_name}' not found in evaluation results. "
        f"Available keys: {available_keys}"
    )


def train_once(
    workload: spec.Workload,
    workload_name: str,
    global_batch_size: int,
    data_dir: str,
    hyperparameters: Optional[spec.Hyperparameters],
    rng_seed: int,
    max_global_steps: int,
    trial: Optional[optuna.Trial] = None,
    eval_frequency: int = 500,
) -> float:
    rng = prng.PRNGKey(rng_seed)
    data_rng, opt_init_rng, model_init_rng, rng = prng.split(rng, 4)

    logging.info("Initializing dataset")
    train_input_queue = workload._build_input_queue(
        data_rng=data_rng,
        split="train",
        data_dir=data_dir,
        global_batch_size=global_batch_size,
    )
    
    num_train_examples = workload.num_train_examples
    num_eval_train_examples = workload.num_eval_train_examples
    num_validation_examples = workload.num_validation_examples
    num_test_examples = workload.num_test_examples
    eval_batch_size = workload.eval_batch_size

    logging.info("Initializing model")
    model_params, model_state = workload.init_model_fn(model_init_rng)
    
    logging.info("Initializing optimizer")
    optimizer_state = submission.init_optimizer_state(
        workload, model_params, model_state, hyperparameters, opt_init_rng
    )
    
    update_fn = submission.update_params
    prepare_for_eval = getattr(submission, 'prepare_for_eval', None)
    needs_train_state = 'train_state' in signature(update_fn).parameters

    train_state = {
        'validation_goal_reached': False,
        'test_goal_reached': False,
        'is_time_remaining': True,
        'last_eval_time': 0,
        'training_complete': False,
        'accumulated_submission_time': 0,
        'accumulated_eval_time': 0,
        'accumulated_logging_time': 0,
        'last_step_end_time': time.time(),
    }
    
    eval_results = []
    global_step = 0
    running_loss = 0.0
    running_loss_count = 0
    best_metric_value = float('-inf') if workload_info[workload_name]["maximize"] else float('inf')

    logging.info("Starting training!")
    while global_step < max_global_steps and not train_state['training_complete']:
        step_rng = prng.fold_in(rng, global_step)
        data_select_rng, update_rng, prep_eval_rng, eval_rng = prng.split(step_rng, 4)
        
        try:
            batch = next(train_input_queue)
        except StopIteration:
            logging.info("Rebuilding train input queue")
            data_rng = prng.fold_in(data_rng, global_step)
            train_input_queue = workload._build_input_queue(
                data_rng=data_rng,
                split="train",
                data_dir=data_dir,
                global_batch_size=global_batch_size,
            )
            batch = next(train_input_queue)

        try:
            update_start_time = time.time()
            update_kwargs = {
                'workload': workload,
                'current_param_container': model_params,
                'current_params_types': workload.model_params_types,
                'model_state': model_state,
                'hyperparameters': hyperparameters,
                'batch': batch,
                'loss_type': workload.loss_type,
                'optimizer_state': optimizer_state,
                'eval_results': eval_results,
                'global_step': global_step,
                'rng': update_rng,
            }
            if needs_train_state:
                update_kwargs['train_state'] = MappingProxyType(train_state)
            
            optimizer_state, model_params, model_state = update_fn(**update_kwargs)
            
            try:
                if hasattr(batch, 'get') and 'targets' in batch:
                    loss = float(workload.loss_fn(model_params, model_state, batch['inputs'], batch['targets'], hyperparameters))
                    running_loss += loss
                    running_loss_count += 1
            except Exception as e:
                logging.warning(f"Could not extract loss: {str(e)}")
                
        except spec.TrainingCompleteError:
            train_state['training_complete'] = True
            break
            
        global_step += 1
        step_end_time = time.time()
        step_duration = step_end_time - train_state['last_step_end_time']
        train_state['accumulated_submission_time'] += step_duration
        train_state['last_step_end_time'] = step_end_time
        
        if global_step % 100 == 0:
            avg_loss = running_loss / max(running_loss_count, 1)
            elapsed_time = train_state['accumulated_submission_time']
            steps_per_sec = global_step / max(elapsed_time, 1)
            
            logging.info(
                f"Step {global_step}/{max_global_steps} | "
                f"Time: {elapsed_time:.1f}s | "
                f"Loss: {avg_loss:.6f} | "
                f"Speed: {steps_per_sec:.1f} steps/s"
            )
            running_loss = 0.0
            running_loss_count = 0
        
        if trial is not None and global_step % eval_frequency == 0:
            logging.info(f"Performing intermediate evaluation at step {global_step}")
            eval_start_time = time.time()
            
            if prepare_for_eval is not None:
                logging.info("Preparing for evaluation")
                optimizer_state, model_params, model_state = prepare_for_eval(
                    workload=workload,
                    current_param_container=model_params,
                    current_params_types=workload.model_params_types,
                    model_state=model_state,
                    hyperparameters=hyperparameters,
                    loss_type=workload.loss_type,
                    optimizer_state=optimizer_state,
                    eval_results=eval_results,
                    global_step=global_step,
                    rng=prep_eval_rng,
                )
            
            validation_result = workload.eval_model(
                eval_batch_size,
                model_params,
                model_state,
                eval_rng,
                data_dir,
                FLAGS.imagenet_v2_data_dir,
                global_step,
            )
            
            eval_end_time = time.time()
            train_state['accumulated_eval_time'] += (eval_end_time - eval_start_time)
            
            metric_name = workload_info[workload_name]["metric_name"]
            try:
                metric_value = get_metric_value(validation_result, metric_name)
            except KeyError as e:
                logging.error(f"Metric extraction failed: {str(e)}")
                metric_value = float('-inf') if workload_info[workload_name]["maximize"] else float('inf')
            
            trial.report(metric_value, global_step)
            
            if trial.should_prune():
                logging.info(f"Trial pruned at step {global_step} with value: {metric_value}")
                raise optuna.exceptions.TrialPruned()
            
            logging.info(f"Intermediate eval @ step {global_step}:")
            logging.info(f"  {metric_name}: {metric_value:.4f}")
            
            is_better = metric_value > best_metric_value if workload_info[workload_name]["maximize"] else metric_value < best_metric_value
            if is_better:
                best_metric_value = metric_value
                logging.info("  New best!")
            
            eval_results.append((global_step, validation_result))

    if prepare_for_eval is not None:
        logging.info("Preparing for final evaluation")
        eval_prep_start_time = time.time()
        optimizer_state, model_params, model_state = prepare_for_eval(
            workload=workload,
            current_param_container=model_params,
            current_params_types=workload.model_params_types,
            model_state=model_state,
            hyperparameters=hyperparameters,
            loss_type=workload.loss_type,
            optimizer_state=optimizer_state,
            eval_results=eval_results,
            global_step=global_step,
            rng=prep_eval_rng,
        )
        eval_prep_end_time = time.time()
        train_state['accumulated_submission_time'] += (eval_prep_end_time - eval_prep_start_time)
    
    logging.info("Running final evaluation")
    eval_start_time = time.time()
    validation_result = workload.eval_model(
        eval_batch_size,
        model_params,
        model_state,
        eval_rng,
        data_dir,
        FLAGS.imagenet_v2_data_dir,
        global_step,
    )
    eval_end_time = time.time()
    train_state['accumulated_eval_time'] += (eval_end_time - eval_start_time)
    
    metric_name = workload_info[workload_name]["metric_name"]
    try:
        metric_value = get_metric_value(validation_result, metric_name)
    except KeyError as e:
        logging.error(f"Metric extraction failed: {str(e)}")
        metric_value = float('-inf') if workload_info[workload_name]["maximize"] else float('inf')
    
    logging.info(f"Training completed in {train_state['accumulated_submission_time']:.1f}s")
    logging.info(f"Final {metric_name}: {metric_value:.4f}")

    return metric_value


def objective(
    trial: optuna.Trial,
    workload: spec.Workload,
    workload_name: str,
    data_dir: str,
    step_hint_factor: float,
) -> float:
    hyperparameters = get_tuned_hyperparameters(trial)
    max_steps = int(workload_info[workload_name]["step_hint"] * step_hint_factor)
    seed = FLAGS.rng_seed
    
    logging.info(f"\nStarting trial with hyperparameters:")
    for key, value in trial.params.items():
        logging.info(f"  {key}: {value}")
    
    eval_frequency = max(500, max_steps // 3)  
    logging.info(f"Will evaluate at steps: {[eval_frequency, 2*eval_frequency, max_steps]}")
    
    try:
        metric_value = train_once(
            workload,
            workload_name,
            submission.get_batch_size(workload_name),
            data_dir,
            hyperparameters,
            seed,
            max_global_steps=max_steps,
            trial=trial,
            eval_frequency=eval_frequency
        )
        
        logging.info(f"Trial completed with {workload_info[workload_name]['metric_name']}: {metric_value}")
        return metric_value
    except optuna.exceptions.TrialPruned:
        logging.info("Trial was pruned by Optuna")
        raise
    except Exception as e:
        logging.error(f"Error during trial: {str(e)}")
        import traceback
        traceback.print_exc()
        return float('-inf') if workload_info[workload_name]["maximize"] else float('inf')


def main(_):
    logging.info(f"\n🚀 Starting tuning with settings:")
    logging.info(f"   Workload: {FLAGS.workload}")
    logging.info(f"   Data dir: {FLAGS.data_dir}")
    logging.info(f"   Output dir: {FLAGS.experiment_dir}")
    logging.info(f"   Step hint factor: {FLAGS.step_hint_factor}")
    logging.info(f"   Trials: {FLAGS.num_tuning_trials}\n")

    os.makedirs(FLAGS.experiment_dir, exist_ok=True)

    if FLAGS.workload not in workload_info:
        raise ValueError(f"❌ Unknown workload: {FLAGS.workload}")

    if FLAGS.workload not in workloads.WORKLOADS:
        raise ValueError(f"Workload {FLAGS.workload} not found in workloads.WORKLOADS")
    workload_metadata = workloads.WORKLOADS[FLAGS.workload]
    workload_path = os.path.join(
        workloads.BASE_WORKLOADS_DIR,
        workload_metadata['workload_path'] + '_jax',
        'workload.py'
    )
    workload_init_kwargs = {}
    workload = workloads.import_workload(
        workload_path=workload_path,
        workload_class_name=workload_metadata['workload_class_name'],
        workload_init_kwargs=workload_init_kwargs
    )
    logging.info(f"✅ Loaded {FLAGS.workload} workload\n")

    study_name = f"kron_tuning_{FLAGS.workload}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize" if workload_info[FLAGS.workload]["maximize"] else "minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200),
        load_if_exists=True,
    )

    wandb_callback = None
    if FLAGS.use_wandb:
        wandb_callback = WeightsAndBiasesCallback(
            metric_name=workload_info[FLAGS.workload]["target_metric_name"],
            wandb_kwargs={
                "project": f"algoperf-{FLAGS.workload}",
                "config": {
                    "workload": FLAGS.workload,
                    "step_hint_factor": FLAGS.step_hint_factor,
                    "num_trials": FLAGS.num_tuning_trials,
                    "rng_seed": FLAGS.rng_seed,
                },
                "tags": ["kron", "tpu", FLAGS.workload],
                "reinit": True,
            },
        )

    callbacks = [wandb_callback] if wandb_callback else []
    
    study.optimize(
        lambda t: objective(t, workload, FLAGS.workload, FLAGS.data_dir, FLAGS.step_hint_factor),
        n_trials=FLAGS.num_tuning_trials,
        callbacks=callbacks,
        gc_after_trial=True,
    )

    logging.info("Best trial:")
    logging.info(f"  Value: {study.best_trial.value}")
    logging.info("  Params: ")
    for key, value in study.best_trial.params.items():
        logging.info(f"    {key}: {value}")


if __name__ == "__main__":
    app.run(main)
