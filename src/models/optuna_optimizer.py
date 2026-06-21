"""
Bayesian hyperparameter optimization with Optuna (TPE sampler).

WHY THIS MATTERS FOR THE PROJECT
--------------------------------
The proposal promises "Bayesian Optimization", but random/grid search are NOT
Bayesian - they ignore everything learned from previous trials. Optuna's default
sampler is the Tree-structured Parzen Estimator (TPE), a genuine Bayesian method.

HOW TPE WORKS (one-paragraph viva answer)
-----------------------------------------
After a few random trials, TPE splits past trials into "good" (top quantile of
the objective) and "bad" groups. It fits two probability densities over the
hyperparameters: l(x) for good trials and g(x) for bad trials. The next trial
maximises the ratio l(x)/g(x) - i.e. it samples hyperparameters that were common
among good configs and rare among bad ones. This concentrates the search budget
where improvement is likely, so TPE typically reaches a better score than random
search in far fewer trials. It also handles mixed integer/float/categorical
spaces and conditional parameters, which Gaussian-Process BO struggles with.

BONUS FOR THE REPORT
--------------------
Optuna records every trial, giving you (a) an optimisation-history curve and
(b) hyperparameter-importance scores (fANOVA) - both excellent figures that show
methodological rigour.
"""

import logging

import numpy as np
from sklearn.model_selection import cross_val_score

logger = logging.getLogger("streamflow_automl.models.optuna_optimizer")


class OptunaOptimizer:
    """Bayesian hyperparameter search for a single model using Optuna TPE."""

    def __init__(self, nse_scorer, random_state: int = 42):
        self.nse_scorer = nse_scorer
        self.random_state = random_state
        self.study = None

    def optimize(self, model_name, model_class, base_params, search_space,
                 X, y, cv, n_trials, nn_param_processor=None):
        """
        Run a TPE study and return (best_params, best_score, study).

        search_space entries follow the YAML schema:
          {type: int|float|categorical, low/high/choices, log: bool}
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = dict(base_params)
            for name, spec in search_space.items():
                params[name] = self._suggest(trial, name, spec)

            if nn_param_processor is not None:
                params = nn_param_processor(params)

            try:
                model = model_class(**params)
                scores = cross_val_score(
                    model, X, y, cv=cv, scoring=self.nse_scorer, n_jobs=1
                )
                return float(np.mean(scores))
            except Exception as exc:  # a bad hyperparameter combo should not crash the study
                logger.debug(f"Trial failed for {model_name}: {exc}")
                return -1e9

        # TPESampler with a fixed seed -> reproducible search (important for a report)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.study = study
        # rebuild the winning parameter dict in the same (decoded) form
        best_params = {}
        for name, spec in search_space.items():
            best_params[name] = self._decode_best(study.best_params, name, spec)

        return best_params, study.best_value, study

    @staticmethod
    def _suggest(trial, name, spec):
        """Ask Optuna for one hyperparameter value given its spec."""
        kind = spec["type"]
        if kind == "int":
            return trial.suggest_int(name, spec["low"], spec["high"])
        if kind == "float":
            return trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        if kind == "categorical":
            choices = spec["choices"]
            # Optuna categoricals must be hashable; list choices (e.g. NN layer
            # configs) are encoded as their index and decoded after suggestion.
            if any(isinstance(c, (list, tuple)) for c in choices):
                idx = trial.suggest_categorical(name, list(range(len(choices))))
                return choices[idx]
            return trial.suggest_categorical(name, choices)
        raise ValueError(f"Unknown hyperparameter type: {kind}")

    @staticmethod
    def _decode_best(best_params, name, spec):
        """Translate Optuna's stored best value back to the usable form."""
        raw = best_params[name]
        if spec["type"] == "categorical":
            choices = spec["choices"]
            if any(isinstance(c, (list, tuple)) for c in choices):
                return choices[raw]
        return raw

    def get_param_importances(self) -> dict:
        """fANOVA hyperparameter importances - a great figure for the report."""
        if self.study is None:
            return {}
        try:
            import optuna
            return optuna.importance.get_param_importances(self.study)
        except Exception:
            return {}
