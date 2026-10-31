from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np


class BaseDetector(ABC):
    def __init__(self, name: str, approach: str):
        self.name = name
        self.approach = approach
        self.model: Any = None

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> "BaseDetector":
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass


class LogisticRegressionDetector(BaseDetector):
    def __init__(self, max_iter: int = 1000):
        super().__init__(name="Regresión Logística", approach="Tabular")
        self.max_iter = max_iter

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> "LogisticRegressionDetector":
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(class_weight="balanced", max_iter=self.max_iter)
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class KerasModelMixin:
    @staticmethod
    def build_callbacks(patience: int = 3) -> List[Any]:
        from tensorflow import keras

        return [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
            ),
        ]


class MLPDetector(BaseDetector, KerasModelMixin):
    def __init__(
        self,
        name: str,
        params: Dict[str, Union[int, float]],
        epochs: int = 30,
        patience: int = 3,
    ):
        super().__init__(name=name, approach="Tabular")
        self.params = dict(params)
        self.epochs = epochs
        self.patience = patience

    def _build_network(self, n_features: int) -> Any:
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([layers.Input((n_features,))])
        for _ in range(int(self.params["n_layers"])):
            model.add(layers.Dense(int(self.params["n_units"]), activation="relu"))
            model.add(layers.Dropout(float(self.params["dropout_rate"])))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=keras.optimizers.Adam(float(self.params["lr"])),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc")],
        )
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> "MLPDetector":
        self.model = self._build_network(X_train.shape[1])
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=int(self.params["batch_size"]),
            class_weight=class_weight,
            callbacks=self.build_callbacks(self.patience),
            verbose=0,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, batch_size=4096, verbose=0).ravel()


class OptunaMLPTuner:
    def __init__(self, trials: int, seed: int, tune_epochs: int = 10):
        self.trials = trials
        self.seed = seed
        self.tune_epochs = tune_epochs

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weight: Dict[int, float],
    ) -> Dict[str, Union[int, float]]:
        import optuna
        from sklearn.metrics import f1_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: Any) -> float:
            params = {
                "n_layers": trial.suggest_int("n_layers", 2, 4),
                "n_units": trial.suggest_categorical("n_units", [64, 128, 256]),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            }
            detector = MLPDetector(name="MLP_Trial", params=params, epochs=self.tune_epochs)
            detector.fit(X_train, y_train, X_val, y_val, class_weight)
            preds = (detector.predict_proba(X_val) >= 0.5).astype(int)
            return float(f1_score(y_val, preds))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(objective, n_trials=self.trials)
        print("Mejores hiperparámetros Optuna:", study.best_params)
        return study.best_params


class LightGBMDetector(BaseDetector):
    def __init__(self, seed: int = 42, use_gpu: bool = False):
        super().__init__(name="LightGBM", approach="Tabular")
        self.seed = seed
        self.use_gpu = use_gpu

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> "LightGBMDetector":
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
            device="gpu" if self.use_gpu else "cpu",
        )
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class CNNLSTMDetector(BaseDetector, KerasModelMixin):
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        epochs: int = 30,
        batch_size: int = 512,
        patience: int = 4,
    ):
        super().__init__(name="CNN-LSTM", approach="Secuencial")
        self.seq_len = seq_len
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

    def _build_network(self) -> Any:
        from tensorflow import keras
        from tensorflow.keras import layers

        inp = layers.Input((self.seq_len, self.n_features))
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc")],
        )
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> "CNNLSTMDetector":
        self.model = self._build_network()
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=self.build_callbacks(self.patience),
            verbose=0,
        )
        return self

    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        return self.model.predict(X, batch_size=batch_size, verbose=0).ravel()
