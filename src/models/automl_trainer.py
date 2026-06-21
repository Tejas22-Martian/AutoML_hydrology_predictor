"""
AutoML trainer with Bayesian Optimization for hydrological model selection.

Trains and tunes multiple model types (Random Forest, XGBoost, LightGBM, SVM,
Neural Network), selects the best performer, and returns it with full metadata.
"""

import logging
import time
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from src.models.metrics import nash_sutcliffe_efficiency
from src.utils.helpers import load_config

logger = logging.getLogger("streamflow_automl.models.automl_trainer")


class AutoMLTrainer:
    """Bayesian Optimization-based AutoML for streamflow prediction models."""

    MODEL_REGISTRY = {
        "random_forest": RandomForestRegressor,
        "xgboost": None,
        "lightgbm": None,
        "svm": SVR,
        "neural_network": MLPRegressor,
        "lstm": None,  # registered at runtime if PyTorch is installed
    }

    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None):
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = config or {}

        self.model_configs = self.config.get("models", {})
        self.best_model = None
        self.best_score = -np.inf
        self.best_model_name = None
        self.results = {}
        self.studies = {}  # Optuna study per model (for history / importance plots)
        self._register_optional_models()

    def _register_optional_models(self):
        """Register XGBoost and LightGBM if available."""
        try:
            from xgboost import XGBRegressor
            self.MODEL_REGISTRY["xgboost"] = XGBRegressor
        except ImportError:
            logger.warning("XGBoost not available, skipping")

        try:
            from lightgbm import LGBMRegressor
            self.MODEL_REGISTRY["lightgbm"] = LGBMRegressor
        except ImportError:
            logger.warning("LightGBM not available, skipping")

        # LSTM needs PyTorch; register only if it imported successfully so the
        # rest of the pipeline still runs on machines without a deep-learning stack
        from src.models.lstm_model import torch_available, LSTMRegressor
        if torch_available():
            self.MODEL_REGISTRY["lstm"] = LSTMRegressor
        else:
            logger.warning("PyTorch not available, LSTM disabled")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 50,
        cv_folds: int = 5,
    ) -> object:
        """
        Run AutoML: optimize each enabled model, return the best one.

        Uses Bayesian Optimization (Optuna TPE sampler) for hyperparameter
        search, with TimeSeriesSplit cross-validation to respect temporal
        ordering (training folds always precede validation folds - no leakage).
        """
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        for model_name, model_config in self.model_configs.items():
            if not model_config.get("enabled", True):
                continue

            model_class = self.MODEL_REGISTRY.get(model_name)
            if model_class is None:
                logger.warning(f"Model class for '{model_name}' not available, skipping")
                continue

            logger.info(f"Optimizing {model_name}...")
            start_time = time.time()

            try:
                best_params, best_score = self._optimize_model(
                    model_name, model_class, model_config,
                    X_train, y_train, tscv, n_trials
                )

                elapsed = time.time() - start_time
                self.results[model_name] = {
                    "best_params": best_params,
                    "best_score": best_score,
                    "training_time_s": round(elapsed, 2),
                }

                logger.info(
                    f"{model_name}: NSE={best_score:.4f} (took {elapsed:.1f}s)"
                )

                if best_score > self.best_score:
                    self.best_score = best_score
                    self.best_model_name = model_name
                    base_params = model_config.get("base_params", {})
                    final_params = {**base_params, **best_params}
                    if model_name == "neural_network":
                        final_params = self._process_nn_params(final_params)
                    self.best_model = model_class(**final_params)
                    self.best_model.fit(X_train, y_train)

            except Exception as e:
                logger.error(f"Failed to optimize {model_name}: {e}")
                self.results[model_name] = {"error": str(e)}

        if self.best_model is None:
            raise RuntimeError("All model optimizations failed")

        logger.info(
            f"Best model: {self.best_model_name} with NSE={self.best_score:.4f}"
        )
        return self.best_model

    def _optimize_model(
        self,
        model_name: str,
        model_class: type,
        model_config: dict,
        X: np.ndarray,
        y: np.ndarray,
        cv,
        n_trials: int,
    ) -> tuple:
        """
        Optimise one model's hyperparameters.

        Priority order:
          1. Optuna TPE  -> genuine Bayesian Optimization (preferred)
          2. random search -> fallback if Optuna is unavailable
        """
        search_space = model_config.get("search_space", {})
        base_params = model_config.get("base_params", {})
        nn_proc = self._process_nn_params if model_name == "neural_network" else None

        try:
            from src.models.optuna_optimizer import OptunaOptimizer
            optimizer = OptunaOptimizer(self._nse_scorer)
            best_params, best_score, study = optimizer.optimize(
                model_name, model_class, base_params, search_space,
                X, y, cv, n_trials, nn_param_processor=nn_proc,
            )
            self.studies[model_name] = study
            return best_params, best_score
        except ImportError:
            logger.info("Optuna not available, falling back to random search")
            return self._random_search(
                model_name, model_class, base_params, search_space,
                X, y, cv, n_trials
            )

    def _bayesian_optimize(
        self, model_name, model_class, base_params, search_space,
        X, y, cv, n_trials
    ):
        """Bayesian Optimization using the bayes_opt library."""
        from bayes_opt import BayesianOptimization

        continuous_params = {}
        categorical_params = {}

        for param_name, param_spec in search_space.items():
            if param_spec["type"] == "categorical":
                categorical_params[param_name] = param_spec["choices"]
            else:
                continuous_params[param_name] = (param_spec["low"], param_spec["high"])

        best_categorical = {k: v[0] for k, v in categorical_params.items()}

        def objective(**kwargs):
            params = {**base_params, **best_categorical}
            for k, v in kwargs.items():
                spec = search_space[k]
                if spec["type"] == "int":
                    params[k] = int(round(v))
                else:
                    params[k] = v

            if model_name == "neural_network":
                params = self._process_nn_params(params)

            try:
                model = model_class(**params)
                scores = cross_val_score(
                    model, X, y, cv=cv,
                    scoring=self._nse_scorer, n_jobs=1,
                )
                return np.mean(scores)
            except Exception:
                return -1.0

        if not continuous_params:
            model = model_class(**{**base_params, **best_categorical})
            scores = cross_val_score(model, X, y, cv=cv, scoring=self._nse_scorer)
            return best_categorical, np.mean(scores)

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=continuous_params,
            random_state=42,
            verbose=0,
        )
        optimizer.maximize(init_points=min(5, n_trials // 3), n_iter=n_trials)

        best = optimizer.max
        best_params = {}
        for k, v in best["params"].items():
            spec = search_space[k]
            best_params[k] = int(round(v)) if spec["type"] == "int" else v
        best_params.update(best_categorical)

        return best_params, best["target"]

    def _random_search(
        self, model_name, model_class, base_params, search_space,
        X, y, cv, n_trials
    ):
        """Fallback random search when bayes_opt is unavailable."""
        rng = np.random.default_rng(42)
        best_score = -np.inf
        best_params = {}

        for trial in range(min(n_trials, 30)):
            params = dict(base_params)
            for param_name, spec in search_space.items():
                if spec["type"] == "int":
                    params[param_name] = int(rng.integers(spec["low"], spec["high"]))
                elif spec["type"] == "float":
                    if spec.get("log", False):
                        log_val = rng.uniform(
                            np.log(spec["low"]), np.log(spec["high"])
                        )
                        params[param_name] = np.exp(log_val)
                    else:
                        params[param_name] = rng.uniform(spec["low"], spec["high"])
                elif spec["type"] == "categorical":
                    # pick by INDEX so list-valued choices (e.g. NN layer
                    # configs like [128, 64]) don't trip numpy's ragged-array
                    # error - this was the cause of the neural_network crash.
                    choices = spec["choices"]
                    params[param_name] = choices[int(rng.integers(0, len(choices)))]

            if model_name == "neural_network":
                params = self._process_nn_params(params)

            try:
                model = model_class(**params)
                scores = cross_val_score(
                    model, X, y, cv=cv, scoring=self._nse_scorer, n_jobs=1,
                )
                score = np.mean(scores)
                if score > best_score:
                    best_score = score
                    best_params = {
                        k: v for k, v in params.items() if k not in base_params
                    }
            except Exception:
                continue

        return best_params, best_score

    @staticmethod
    def _process_nn_params(params: dict) -> dict:
        """Convert neural network config to MLPRegressor parameters."""
        params = dict(params)
        if "hidden_layer_config" in params:
            params["hidden_layer_sizes"] = tuple(params.pop("hidden_layer_config"))
        if "batch_size" in params:
            params["batch_size"] = int(params["batch_size"])
        return params

    @staticmethod
    def _nse_scorer(estimator, X, y):
        """Custom scorer returning Nash-Sutcliffe Efficiency."""
        y_pred = estimator.predict(X)
        return nash_sutcliffe_efficiency(y, y_pred)

    def save_model(self, filepath: str) -> None:
        """Save the best trained model to disk."""
        if self.best_model is None:
            raise RuntimeError("No model trained yet")
        joblib.dump(
            {
                "model": self.best_model,
                "model_name": self.best_model_name,
                "score": self.best_score,
                "results": self.results,
            },
            filepath,
        )
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> dict:
        """Load a saved model from disk."""
        return joblib.load(filepath)
