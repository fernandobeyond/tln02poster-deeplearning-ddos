import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd


class ReportGenerator:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def print_section(title: str) -> None:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)

    @classmethod
    def to_native_python(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: cls.to_native_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls.to_native_python(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    def save_csv(self, df: pd.DataFrame, filename: str) -> None:
        df.to_csv(os.path.join(self.out_dir, filename), index=False)

    def save_json(self, results: Dict[str, Any], filename: str = "resultados.json") -> None:
        path = os.path.join(self.out_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_native_python(results), fh, ensure_ascii=False, indent=2)

    def plot_f1_comparison(self, table_df: pd.DataFrame, filename: str = "f1_modelos.png") -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(table_df["modelo"], table_df["f1"], color=(0.231, 0.435, 0.714))
            for i, v in enumerate(table_df["f1"]):
                ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("F1-Score (clase ataque)")
            ax.set_title("Comparativa de modelos")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, filename), dpi=200)
            plt.close(fig)
        except ImportError:
            pass
