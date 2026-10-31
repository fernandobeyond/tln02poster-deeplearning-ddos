from typing import Any, Dict, List

import numpy as np
import pandas as pd


class MetricsEvaluator:
    def __init__(self):
        self.model_metrics: Dict[str, Dict[str, Any]] = {}
        self.comparison_rows: List[Dict[str, Any]] = []

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        has_both_classes = len(np.unique(y_true)) > 1
        return {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both_classes else 0.0,
            "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else 0.0,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "fpr": float(fp / max(fp + tn, 1)),
        }

    def register_model(
        self,
        name: str,
        approach: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, Any]:
        metrics = self.compute_metrics(y_true, y_prob)
        self.model_metrics[name] = metrics
        self.comparison_rows.append(
            {
                "modelo": name,
                "enfoque": approach,
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "accuracy": metrics["accuracy"],
            }
        )
        print(
            f"  {name:<24} F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
            f"R={metrics['recall']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}  Acc={metrics['accuracy']:.4f}"
        )
        return metrics

    def get_comparison_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.comparison_rows).sort_values("f1")

    def print_confusion_matrix(self, model_name: str, n_test: int) -> None:
        cm = self.model_metrics[model_name]
        print(f"  Verdadero negativo : {cm['tn']:,}")
        print(f"  Falso positivo     : {cm['fp']:,}")
        print(f"  Falso negativo     : {cm['fn']:,}")
        print(f"  Verdadero positivo : {cm['tp']:,}")
        print(f"  Flujos de prueba   : {n_test:,}")
        print(f"  FPR                : {cm['fpr']:.4%}")
