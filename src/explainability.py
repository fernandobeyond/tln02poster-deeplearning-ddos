from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.models import CNNLSTMDetector, LightGBMDetector


class ExplainabilityAnalyzer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.importance_rows: List[Dict[str, Any]] = []

    def compute_permutation_importance(
        self,
        detector: LightGBMDetector,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> List[Tuple[str, float]]:
        from sklearn.inspection import permutation_importance

        rng = np.random.default_rng(self.config.seed)
        sub = rng.choice(len(y_test), min(100_000, len(y_test)), replace=False)
        pi = permutation_importance(
            detector.model,
            X_test[sub],
            y_test[sub],
            scoring="f1",
            n_repeats=3,
            random_state=self.config.seed,
            n_jobs=1,
        )
        order = np.argsort(pi.importances_mean)[::-1]
        print("LightGBM - Permutation Importance (caída de F1):")
        for r, j in enumerate(order[:3], 1):
            print(f"  {r}. {self.config.features[j]:<28} {pi.importances_mean[j]:.4f}")
        for r, j in enumerate(order, 1):
            self.importance_rows.append(
                {
                    "modelo": "LightGBM",
                    "metodo": "permutation",
                    "rango": r,
                    "atributo": self.config.features[j],
                    "valor": float(pi.importances_mean[j]),
                }
            )
        return [(self.config.features[j], float(pi.importances_mean[j])) for j in order[:3]]

    def compute_kernel_shap(
        self,
        detector: CNNLSTMDetector,
        X_seq_train: np.ndarray,
        X_seq_test: np.ndarray,
    ) -> List[Tuple[str, float]]:
        if self.config.skip_shap:
            return []
        try:
            import shap

            rng = np.random.default_rng(self.config.seed)
            seq_len = self.config.seq_len
            n_feat = len(self.config.features)
            bg = X_seq_train[rng.choice(len(X_seq_train), min(50, len(X_seq_train)), replace=False)]
            ex = X_seq_test[
                rng.choice(len(X_seq_test), min(self.config.shap_samples, len(X_seq_test)), replace=False)
            ]

            def predict_fn(a: np.ndarray) -> np.ndarray:
                return detector.predict_proba(a.reshape(-1, seq_len, n_feat), batch_size=2048)

            explainer = shap.KernelExplainer(predict_fn, bg.reshape(len(bg), -1))
            sv = explainer.shap_values(ex.reshape(len(ex), -1), nsamples=200)
            sv_raw = sv[0] if isinstance(sv, list) else sv
            sv_arr = np.asarray(sv_raw).reshape(len(ex), seq_len, n_feat)
            shap_feat = np.abs(sv_arr).mean(axis=(0, 1))
            order_s = np.argsort(shap_feat)[::-1]
            print("CNN-LSTM - KernelSHAP (|SHAP| medio, agregado en los 20 pasos):")
            for r, j in enumerate(order_s[:3], 1):
                print(f"  {r}. {self.config.features[j]:<28} {shap_feat[j]:.4f}")
            for r, j in enumerate(order_s, 1):
                self.importance_rows.append(
                    {
                        "modelo": "CNN-LSTM",
                        "metodo": "kernelshap",
                        "rango": r,
                        "atributo": self.config.features[j],
                        "valor": float(shap_feat[j]),
                    }
                )
            return [(self.config.features[j], float(shap_feat[j])) for j in order_s[:3]]
        except ImportError:
            print("shap no está instalado: se omite KernelSHAP (pip install shap).")
            return []

    def get_importance_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.importance_rows)
