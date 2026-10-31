from typing import Any, Dict, List

import numpy as np
import pandas as pd


class ConfidenceCalibrator:
    def __init__(self, ece_bins: int = 10):
        self.ece_bins = ece_bins
        self.calibration_rows: List[Dict[str, Any]] = []
        self.optimal_temperature: float = 1.0

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        clipped = np.clip(p, eps, 1.0 - eps)
        return np.log(clipped / (1.0 - clipped))

    def compute_ece(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        edges = np.linspace(0, 1, self.ece_bins + 1)
        idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, self.ece_bins - 1)
        ece = 0.0
        for b in range(self.ece_bins):
            m = idx == b
            if m.any():
                ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
        return float(ece)

    def compute_calibration_report(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import brier_score_loss

        return {
            "ece": self.compute_ece(y_true, y_prob),
            "brier": float(brier_score_loss(y_true, y_prob)),
        }

    def register_calibration(self, model_name: str, method: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        report = self.compute_calibration_report(y_true, y_prob)
        self.calibration_rows.append({"modelo": model_name, "metodo": method, **report})
        print(f"  {model_name:<10} {method:<22} ECE={report['ece']:.6f}  Brier={report['brier']:.6f}")

    def calibrate_lightgbm(
        self,
        p_val: np.ndarray,
        y_val: np.ndarray,
        p_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        self.register_calibration("LightGBM", "original", y_test, p_test)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, y_val)
        self.register_calibration("LightGBM", "isotónica", y_test, iso.predict(p_test))
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(self.logit(p_val).reshape(-1, 1), y_val)
        p_platt = platt.predict_proba(self.logit(p_test).reshape(-1, 1))[:, 1]
        self.register_calibration("LightGBM", "sigmoide (Platt)", y_test, p_platt)

    @classmethod
    def fit_temperature_scaling(cls, logits_val: np.ndarray, y_val: np.ndarray) -> float:
        from scipy.optimize import minimize_scalar

        def nll(T: float) -> float:
            p = np.clip(cls.sigmoid(logits_val / T), 1e-7, 1.0 - 1e-7)
            return float(-np.mean(y_val * np.log(p) + (1.0 - y_val) * np.log(1.0 - p)))

        return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)

    def calibrate_cnn_lstm(
        self,
        p_val: np.ndarray,
        y_val: np.ndarray,
        p_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        self.register_calibration("CNN-LSTM", "original", y_test, p_test)
        T = self.fit_temperature_scaling(self.logit(p_val), y_val)
        self.optimal_temperature = T
        p_temp = self.sigmoid(self.logit(p_test) / T)
        self.register_calibration("CNN-LSTM", f"temperature (T={T:.3f})", y_test, p_temp)
        return T

    def get_calibration_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.calibration_rows)
