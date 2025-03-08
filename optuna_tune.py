import os
import json
import optuna
import tempfile
import uuid
import time
from typing import Dict, Any, List
from algoperf.workloads import workloads
import submission_runner
from absl import flags


FIXED_SEED = 42

class OptunaMetricsLogger:
    """Logger class for tracking metrics during Optuna trials."""
    
    def __init__(self):
        self.validation_map: float = -1.0
        self.current_trial_metrics: List[Dict[str, Any]] = []

    def append_scalar_metrics(self, metrics: Dict[str, Any], **kwargs) -> None:
        """Append metrics from current evaluation step.
        
        Args:
            metrics: Dictionary containing metric names and values
            **kwargs: Additional keyword arguments
        """
        if 'validation/mean_average_precision' in metrics:
            self.validation_map = max(self.validation_map, metrics['validation/mean_average_precision'])
        self.current_trial_metrics.append(metrics)

    def finish(self) -> None:
        """Cleanup method called at the end of logging."""
        pass

def create_search_space(trial: optuna.Trial) -> Dict[str, Dict[str, List[float]]]:
    """Create the hyperparameter search space for optimization.
    
    Args:
        trial: Optuna trial object for suggesting hyperparameters
        
    Returns:
        Dictionary containing the search space configuration
    """
    return {
        "learning_rate": {"feasible_points": [trial.suggest_float("learning_rate", 0.0002, 0.002, log=True)]},
        "weight_decay": {"feasible_points": [trial.suggest_float("weight_decay", 1e-4, 1.0, log=True)]},
        "preconditioner_lr": {"feasible_points": [trial.suggest_float("preconditioner_lr", 0.1, 1.5)]},
        "label_smoothing": {"feasible_points": [trial.suggest_float("label_smoothing", 0.0, 0.25)]},
        "dropout_rate": {"feasible_points": [trial.suggest_float("dropout_rate", 0.0, 0.5)]},
        "warmup_factor": {"feasible_points": [0.0]},
        "step_hint_factor": {"feasible_points": [0.5]},
    }

def run_trial(trial: optuna.Trial, hyperparameters: Dict[str, Any]) -> float:
    """Run a single optimization trial with given hyperparameters.
    
    Args:
        trial: Optuna trial object
        hyperparameters: Dictionary of hyperparameters to test
        
    Returns:
        Validation MAP score for the trial
        
    Raises:
        Exception: If trial execution fails
    """
    flags.FLAGS.unparse_flags()
    flags.FLAGS(['submission_runner.py'])
    
    uid = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    exp_dir = os.path.expanduser(f"~/algorithmic-efficiency/experiments/optuna_{uid}")
    
    flags.FLAGS.workload = "ogbg"
    flags.FLAGS.framework = "jax"
    flags.FLAGS.submission_path = "submissions/external_tuning/kron/submission.py"
    flags.FLAGS.tuning_ruleset = "external"
    flags.FLAGS.num_tuning_trials = 1
    flags.FLAGS.data_dir = "/dev/shm/ogbg"
    flags.FLAGS.save_checkpoints = False
    flags.FLAGS.use_wandb = False
    flags.FLAGS.rng_seed = FIXED_SEED
    flags.FLAGS.experiment_dir = exp_dir
    flags.FLAGS.experiment_name = f"trial_{trial.number}"
    
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(hyperparameters, temp_file)
        temp_file.close()
        flags.FLAGS.tuning_search_space = temp_file.name

        metrics_logger = OptunaMetricsLogger()
        original_import_workload = workloads.import_workload
        
        def patched_import_workload(*args, **kwargs):
            workload = original_import_workload(*args, **kwargs)
            workload.metrics_logger = metrics_logger
            return workload
        
        try:
            workloads.import_workload = patched_import_workload
            submission_runner.main([])
        finally:
            workloads.import_workload = original_import_workload
            
        return metrics_logger.validation_map
            
    except Exception as e:
        raise RuntimeError(f"Trial execution failed: {str(e)}") from e
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)

def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Validation MAP score
        
    Raises:
        optuna.TrialPruned: If no valid MAP score is recorded
    """
    hyperparameters = create_search_space(trial)
    try:
        map_score = run_trial(trial, hyperparameters)
        if map_score <= 0:
            raise optuna.TrialPruned("No valid MAP score recorded")
        return map_score
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        raise

def run_optimization(n_trials: int = 100) -> None:
    """Run the full optimization process.
    
    Args:
        n_trials: Number of trials to run
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=FIXED_SEED),
        pruner=optuna.pruners.MedianPruner()
    )
    
    try:
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
        raise
    finally:
        print(f"\nBest trial (MAP: {study.best_trial.value}):")
        for k, v in study.best_trial.params.items():
            print(f"  {k:20}: {v}")
        
        study_dir = os.path.expanduser(
            f"~/algorithmic-efficiency/experiments/optuna_final_{int(time.time())}")
        os.makedirs(study_dir, exist_ok=True)
        
        try:
            with open(os.path.join(study_dir, "best_params.json"), 'w') as f:
                json.dump(study.best_params, f, indent=2)
            
            study_path = os.path.join(study_dir, "study.pkl")
            optuna.study.save_study(study, study_path)
            print(f"\nStudy saved to: {study_dir}")
        except Exception as e:
            print(f"Failed to save study results: {str(e)}")

if __name__ == "__main__":
    try:
        run_optimization()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise 