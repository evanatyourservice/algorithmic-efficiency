from typing import Optional, Dict
import os, time, importlib, traceback
from dataclasses import dataclass
from types import MappingProxyType
from inspect import signature
from datetime import datetime
from absl import app, flags, logging
import optuna
from optuna.integration import WeightsAndBiasesCallback
import tensorflow as tf

from algoperf import spec
from algoperf import random_utils as prng
from submissions.external_tuning.kron import submission

tf.config.set_visible_devices([], "GPU")
FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "framework", "jax", enum_values=["jax", "pytorch"], help="Framework for submission."
)
flags.DEFINE_string("data_dir", "~/data", "Path to the data directory")
flags.DEFINE_string(
    "experiment_dir",
    f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "Experiment output directory",
)
flags.DEFINE_integer("num_tuning_trials", 5, "Number of tuning trials")
flags.DEFINE_integer(
    "rng_seed", 42, "Value of rng seed. If None, a random seed will be generated."
)
flags.DEFINE_boolean("use_wandb", True, "Whether to use Weights & Biases logging.")

WORKLOAD_CLASSES = {
    "cifar": {
        "module_name": "algoperf.workloads.cifar.cifar_jax.workload",
        "class_name": "CifarWorkload",
    },
    "criteo1tb": {
        "module_name": "algoperf.workloads.criteo1tb.criteo1tb_jax.workload",
        "class_name": "Criteo1TbDlrmSmallWorkload",
    },
    "fastmri": {
        "module_name": "algoperf.workloads.fastmri.fastmri_jax.workload",
        "class_name": "FastMRIWorkload",
    },
    "imagenet_resnet": {
        "module_name": "algoperf.workloads.imagenet_resnet.imagenet_jax.workload",
        "class_name": "ImagenetResnetWorkload",
    },
    "imagenet_vit": {
        "module_name": "algoperf.workloads.imagenet_vit.imagenet_jax.workload",
        "class_name": "ImagenetVitWorkload",
    },
    "librispeech_conformer": {
        "module_name": "algoperf.workloads.librispeech_conformer.librispeech_jax.workload",
        "class_name": "LibriSpeechConformerWorkload",
    },
    "librispeech_deepspeech": {
        "module_name": "algoperf.workloads.librispeech_deepspeech.librispeech_jax.workload",
        "class_name": "LibriSpeechDeepSpeechWorkload",
    },
    "mnist": {
        "module_name": "algoperf.workloads.mnist.mnist_jax.workload",
        "class_name": "MnistWorkload",
    },
    "ogbg": {
        "module_name": "algoperf.workloads.ogbg.ogbg_jax.workload",
        "class_name": "OgbgWorkload",
    },
    "wmt": {
        "module_name": "algoperf.workloads.wmt.wmt_jax.workload",
        "class_name": "WmtWorkload",
    },
}


@dataclass
class Hyperparameters:
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    preconditioner_lr: float = 0.5
    label_smoothing: float = 0.0
    dropout_rate: float = 0.0


def get_tuned_hyperparameters(trial: optuna.Trial) -> Hyperparameters:
    return Hyperparameters(
        learning_rate=trial.suggest_float("learning_rate", 0.0002, 0.002, log=True),
        weight_decay=trial.suggest_float("weight_decay", 0.0001, 1.0, log=True),
        preconditioner_lr=trial.suggest_float("preconditioner_lr", 0.3, 1.5),
        label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.25),
        dropout_rate=trial.suggest_float("dropout_rate", 0.0, 0.5),
    )


WORKLOAD_INFO = {
    "cifar": {
        "metric_name": "accuracy",
        "maximize": True,
        "step_hint": 4883,
        "target_metric_name": "accuracy",
    },
    "criteo1tb": {
        "metric_name": "loss",
        "maximize": False,
        "step_hint": 10_666,
        "target_metric_name": "loss",
    },
    "fastmri": {
        "metric_name": "ssim",
        "maximize": True,
        "step_hint": 1000,
        "target_metric_name": "ssim",
    },
    "imagenet_resnet": {
        "metric_name": "accuracy",
        "maximize": True,
        "step_hint": 24_541,
        "target_metric_name": "accuracy",
    },
    "imagenet_vit": {
        "metric_name": "accuracy",
        "maximize": True,
        "step_hint": 24_541,
        "target_metric_name": "accuracy",
    },
    "librispeech_conformer": {
        "metric_name": "wer",
        "maximize": False,
        "step_hint": 20_000,
        "target_metric_name": "wer",
    },
    "librispeech_deepspeech": {
        "metric_name": "wer",
        "maximize": False,
        "step_hint": 20_000,
        "target_metric_name": "wer",
    },
    "mnist": {
        "metric_name": "accuracy",
        "maximize": True,
        "step_hint": 109,
        "target_metric_name": "accuracy",
    },
    "ogbg": {
        "metric_name": "mean_average_precision",
        "maximize": True,
        "step_hint": 5_000,
        "target_metric_name": "mean_average_precision",
    },
    "wmt": {
        "metric_name": "bleu",
        "maximize": True,
        "step_hint": 10_000,
        "target_metric_name": "bleu",
    },
}


def get_metric_value(result: Dict[str, float], metric_name: str) -> float:
    if metric_name in result:
        return result[metric_name]
    if f"validation/{metric_name}" in result:
        return result[f"validation/{metric_name}"]
    raise KeyError(
        f"Metric '{metric_name}' not found in evaluation results. Available keys: {', '.join(result.keys())}"
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
    data_rng, opt_init_rng, model_init_rng, train_rng = prng.split(
        prng.PRNGKey(rng_seed), 4
    )
    train_input_queue = workload._build_input_queue(
        data_rng=data_rng,
        split="train",
        data_dir=data_dir,
        global_batch_size=global_batch_size,
    )
    eval_batch_size = workload.eval_batch_size

    try:
        model_params, model_state = workload.init_model_fn(
            model_init_rng,
            dropout_rate=getattr(hyperparameters, "dropout_rate", 0.0),
            aux_dropout_rate=getattr(hyperparameters, "aux_dropout_rate", 0.0),
        )
    except Exception as e:
        logging.error(f"Model initialization failed: {str(e)}")
        raise

    optimizer_state = submission.init_optimizer_state(
        workload, model_params, model_state, hyperparameters, opt_init_rng
    )
    update_fn = submission.update_params
    needs_train_state = "train_state" in signature(update_fn).parameters

    train_state = {
        "validation_goal_reached": False,
        "test_goal_reached": False,
        "is_time_remaining": True,
        "last_eval_time": 0,
        "training_complete": False,
        "accumulated_submission_time": 0,
        "accumulated_eval_time": 0,
        "accumulated_logging_time": 0,
        "last_step_end_time": None,
    }
    step_rng, global_step, eval_results, latest_eval_metrics = train_rng, 0, [], None
    global_start_time = train_state["last_step_end_time"] = last_log_time = time.time()

    while global_step < max_global_steps and not train_state["training_complete"]:
        step_rng = prng.fold_in(train_rng, global_step)
        update_rng = prng.fold_in(step_rng, 1)
        current_time = time.time()

        if current_time - last_log_time > 10:
            logging.info(
                f"Step {global_step}/{max_global_steps} - {(global_step/max_global_steps)*100:.1f}%"
            )
            last_log_time = current_time

        if (
            global_step > 10
            and (current_time - global_start_time) / max(1, global_step) > 60
        ):
            logging.warning("Training is taking too long per step, aborting!")
            break

        try:
            batch_start_time = time.time()
            try:
                batch = next(train_input_queue)
                if time.time() - batch_start_time > 60:
                    logging.warning("Batch loading took more than 60 seconds")
            except StopIteration:
                train_input_queue = workload._build_input_queue(
                    data_rng=prng.fold_in(data_rng, global_step),
                    split="train",
                    data_dir=data_dir,
                    global_batch_size=global_batch_size,
                )
                continue
            except Exception:
                if time.time() - batch_start_time > 120:
                    logging.error("Batch loading timed out, stopping training")
                    break
                continue

            update_kwargs = {
                "workload": workload,
                "current_param_container": model_params,
                "current_params_types": workload.model_params_types,
                "model_state": model_state,
                "hyperparameters": hyperparameters,
                "batch": batch,
                "loss_type": workload.loss_type,
                "optimizer_state": optimizer_state,
                "eval_results": eval_results,
                "global_step": global_step,
                "rng": update_rng,
            }

            if needs_train_state:
                update_kwargs["train_state"] = MappingProxyType(train_state)

            optimizer_state, model_params, model_state = update_fn(**update_kwargs)
        except Exception as e:
            logging.error(f"Error in training step: {str(e)}")
            global_step += 1
            continue

        global_step += 1
        current_time = time.time()
        train_state["accumulated_submission_time"] += (
            current_time - train_state["last_step_end_time"]
        )
        train_state["last_step_end_time"] = current_time

        if global_step % 20 == 0:
            steps_per_sec = global_step / max(
                train_state["accumulated_submission_time"], 1
            )
            logging.info(
                f"Step {global_step}/{max_global_steps} | "
                f"Time: {train_state['accumulated_submission_time']:.1f}s | "
                f"Speed: {steps_per_sec:.1f} steps/s"
            )

        if (global_step % eval_frequency == 0) or (global_step == max_global_steps):
            try:
                eval_start_time = time.time()
                eval_metrics = workload.eval_model(
                    eval_batch_size,
                    model_params,
                    model_state,
                    prng.fold_in(step_rng, global_step),
                    data_dir,
                    None,
                    global_step,
                )
                train_state["accumulated_eval_time"] += time.time() - eval_start_time

                metric_name = WORKLOAD_INFO[workload_name]["metric_name"]
                validation_metric = eval_metrics.get(f"validation/{metric_name}")
                test_metric = eval_metrics.get(f"test/{metric_name}")

                if validation_metric is not None:
                    logging.info(f"Validation {metric_name}: {validation_metric:.6f}")
                if test_metric is not None:
                    logging.info(f"Test {metric_name}: {test_metric:.6f}")

                train_state.update(
                    {
                        "validation_goal_reached": workload.has_reached_validation_target(
                            eval_metrics
                        ),
                        "test_goal_reached": workload.has_reached_test_target(
                            eval_metrics
                        ),
                    }
                )
                latest_eval_metrics = eval_metrics
                eval_results.append((global_step, eval_metrics))

                if trial is not None and validation_metric is not None:
                    trial.report(validation_metric, global_step)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            except Exception as e:
                logging.error(f"Error during evaluation: {str(e)}")

    if latest_eval_metrics:
        try:
            return get_metric_value(
                latest_eval_metrics, WORKLOAD_INFO[workload_name]["metric_name"]
            )
        except Exception as e:
            logging.error(f"Error extracting metric: {str(e)}")
    return float("-inf") if WORKLOAD_INFO[workload_name]["maximize"] else float("inf")


def objective(
    trial: optuna.Trial, workload: spec.Workload, workload_name: str, data_dir: str
) -> float:
    hyperparameters = get_tuned_hyperparameters(trial)
    max_steps = min(
        WORKLOAD_INFO[workload_name]["step_hint"],
        1000 if trial.number < 2 else WORKLOAD_INFO[workload_name]["step_hint"],
    )
    eval_frequency = min(500, max(100, max_steps // 5))
    start_time = time.time()

    try:
        metric_value = train_once(
            workload,
            workload_name,
            submission.get_batch_size(workload_name),
            data_dir,
            hyperparameters,
            FLAGS.rng_seed,
            max_global_steps=max_steps,
            trial=trial,
            eval_frequency=eval_frequency,
        )
        logging.info(
            f"Trial completed in {time.time() - start_time:.1f}s with "
            f"{WORKLOAD_INFO[workload_name]['metric_name']}: {metric_value}"
        )
        return metric_value
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logging.error(f"Error during trial: {str(e)}")
        traceback.print_exc()
        return (
            float("-inf") if WORKLOAD_INFO[workload_name]["maximize"] else float("inf")
        )
    finally:
        if time.time() - start_time > 3600:
            logging.warning("Trial exceeded time limit of 3600s")


def main(_):
    os.makedirs(FLAGS.experiment_dir, exist_ok=True)
    data_dir = os.path.expanduser(FLAGS.data_dir)
    workload_name, num_trials = "ogbg", min(FLAGS.num_tuning_trials, 3)

    module = importlib.import_module(WORKLOAD_CLASSES[workload_name]["module_name"])
    workload = getattr(module, WORKLOAD_CLASSES[workload_name]["class_name"])()

    study = optuna.create_study(
        study_name="ogbg_tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=1, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=200),
    )

    callbacks = []
    if FLAGS.use_wandb:
        callbacks.append(
            WeightsAndBiasesCallback(
                metric_name=WORKLOAD_INFO[workload_name]["target_metric_name"],
                wandb_kwargs={
                    "project": "algoperf-ogbg",
                    "config": {
                        "num_trials": num_trials,
                        "rng_seed": FLAGS.rng_seed,
                        "max_steps": WORKLOAD_INFO[workload_name]["step_hint"],
                    },
                    "tags": ["kron", "ogbg"],
                    "reinit": True,
                },
            )
        )

    try:
        study.optimize(
            lambda trial: objective(
                trial=trial,
                workload=workload,
                workload_name=workload_name,
                data_dir=data_dir,
            ),
            n_trials=num_trials,
            callbacks=callbacks,
            gc_after_trial=True,
            timeout=7200,
        )
    except KeyboardInterrupt:
        logging.info("Study interrupted by user")
    except Exception as e:
        logging.error(f"Study failed: {str(e)}")
        traceback.print_exc()

    if study.best_trial:
        logging.info(f"\nBest trial:\n  Value: {study.best_trial.value}\n  Params:")
        for k, v in study.best_trial.params.items():
            logging.info(f"    {k}: {v}")
    else:
        logging.warning("No successful trials completed")


if __name__ == "__main__":
    app.run(main)
