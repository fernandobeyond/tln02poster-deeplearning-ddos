import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import ExperimentConfig


@dataclass
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_weights: Dict[int, float]


class DataPreprocessor:
    def __init__(self):
        self.imputer: Any = None
        self.scaler: Any = None

    def fit(self, X_fit: np.ndarray) -> "DataPreprocessor":
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        self.imputer = SimpleImputer(strategy="median").fit(X_fit)
        self.scaler = StandardScaler().fit(self.imputer.transform(X_fit))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        imputed = self.imputer.transform(X)
        return self.scaler.transform(imputed).astype(np.float32)


class DatasetManager:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.X: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.y: np.ndarray = np.empty((0,), dtype=np.int8)

    def _extract_file_arrays(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        header_df = pd.read_csv(filepath, nrows=0)
        col_map = {c.strip(): c for c in header_df.columns}

        label_col_stripped = self._resolve_label_column(list(col_map.keys()))
        self._validate_features(list(col_map.keys()))

        needed_raw_cols = [col_map[f] for f in self.config.features] + [col_map[label_col_stripped]]
        df = pd.read_csv(filepath, usecols=needed_raw_cols, low_memory=False)
        df.columns = df.columns.str.strip()

        lab = df[label_col_stripped]
        if pd.api.types.is_numeric_dtype(lab):
            y_part = (pd.to_numeric(lab, errors="coerce").fillna(0).to_numpy(dtype=np.float64) != 0).astype(np.int8)
        else:
            txt = lab.astype(str).str.strip().str.upper()
            y_part = (~txt.isin(["BENIGN", "0", "NORMAL", "0.0"])).to_numpy(dtype=np.int8)

        feat_df = df[self.config.features].apply(pd.to_numeric, errors="coerce")
        X_part = feat_df.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        del df, feat_df
        return X_part, y_part

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        path = self.config.data_path
        files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
        if not files:
            raise FileNotFoundError(f"No se encontraron CSV en: {path}")
        print(f"Cargando {len(files)} archivo(s)...")

        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        for filepath in files:
            X_p, y_p = self._extract_file_arrays(filepath)
            X_parts.append(X_p)
            y_parts.append(y_p)

        X = np.concatenate(X_parts, axis=0) if len(X_parts) > 1 else X_parts[0]
        y = np.concatenate(y_parts, axis=0) if len(y_parts) > 1 else y_parts[0]
        del X_parts, y_parts

        if self.config.sample and self.config.sample < len(X):
            rng = np.random.default_rng(self.config.seed)
            keep = np.sort(rng.choice(len(X), self.config.sample, replace=False))
            X, y = X[keep], y[keep]
            print(f"Submuestreo (orden preservado): {len(X):,} filas")

        self.X, self.y = X, y
        return self.X, self.y

    def _resolve_label_column(self, columns: List[str]) -> str:
        label_col = self.config.label_col
        if label_col == "auto":
            candidates = [c for c in ("Label", "label", "y", "Class") if c in columns]
            if not candidates:
                raise KeyError("No se halló la columna de etiqueta; use --label-col")
            return candidates[0]
        if label_col not in columns:
            raise KeyError(f"Columna de etiqueta '{label_col}' no existe")
        return label_col

    def _validate_features(self, columns: List[str]) -> None:
        missing = [c for c in self.config.features if c not in columns]
        if missing:
            raise KeyError(f"Faltan atributos en el dataset: {missing}")

    @staticmethod
    def _compute_balanced_weights(y_train: np.ndarray) -> Dict[int, float]:
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y_train)
        cw_arr = compute_class_weight("balanced", classes=classes, y=y_train)
        weights = {int(c): float(w) for c, w in zip(classes, cw_arr)}
        weights.setdefault(0, 1.0)
        weights.setdefault(1, 1.0)
        return weights

    def prepare_tabular_splits(self) -> Tuple[DataSplit, Dict[str, int]]:
        from sklearn.model_selection import train_test_split

        n, n_feat = self.X.shape
        n_att = int(self.y.sum())
        print(f"Flujos totales      : {n:,}")
        print(f"Atributos           : {n_feat}")
        print(f"Ataques (clase 1)   : {n_att:,} ({n_att / n:.2%})")
        print(f"Benignos (clase 0)  : {n - n_att:,} ({(n - n_att) / n:.2%})")

        idx_all = np.arange(n)
        i_tr, i_te = train_test_split(
            idx_all,
            test_size=self.config.test_size,
            stratify=self.y,
            random_state=self.config.seed,
        )
        i_tr, i_va = train_test_split(
            i_tr,
            test_size=self.config.val_size_of_train,
            stratify=self.y[i_tr],
            random_state=self.config.seed,
        )
        print(f"Train / Val / Test  : {len(i_tr):,} / {len(i_va):,} / {len(i_te):,}")

        summary = {
            "flujos": n,
            "atributos": n_feat,
            "ataques": n_att,
            "benignos": n - n_att,
            "train": len(i_tr),
            "val": len(i_va),
            "test": len(i_te),
        }

        preprocessor = DataPreprocessor().fit(self.X[i_tr])
        X_tr = preprocessor.transform(self.X[i_tr])
        X_va = preprocessor.transform(self.X[i_va])
        X_te = preprocessor.transform(self.X[i_te])
        y_tr, y_va, y_te = self.y[i_tr], self.y[i_va], self.y[i_te]
        cw = self._compute_balanced_weights(y_tr)

        return DataSplit(X_tr, y_tr, X_va, y_va, X_te, y_te, cw), summary

    @staticmethod
    def build_sequences_index(n: int, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
        ends = np.arange(seq_len - 1, n, stride)
        idx = (ends - seq_len + 1)[:, None] + np.arange(seq_len)[None, :]
        return ends, idx

    def prepare_sequential_splits(self) -> Tuple[DataSplit, Dict[str, int]]:
        from sklearn.model_selection import train_test_split

        n = len(self.X)
        seq_len = self.config.seq_len
        stride = self.config.seq_stride
        ends, widx = self.build_sequences_index(n, seq_len, stride)
        ys = self.y[ends]
        s_all = np.arange(len(ends))

        s_tr, s_te = train_test_split(
            s_all,
            test_size=self.config.test_size,
            stratify=ys,
            random_state=self.config.seed,
        )
        s_tr, s_va = train_test_split(
            s_tr,
            test_size=self.config.val_size_of_train,
            stratify=ys[s_tr],
            random_state=self.config.seed,
        )

        preprocessor = DataPreprocessor().fit(self.X[ends[s_tr]])
        Xs_all = preprocessor.transform(self.X)
        Xseq = Xs_all[widx]
        print(f"Secuencias: {Xseq.shape}  (SEQ_LEN={seq_len}, stride={stride})")

        seq_info = {
            "n_secuencias": int(Xseq.shape[0]),
            "seq_len": seq_len,
            "stride": stride,
        }

        Xs_tr, Xs_va, Xs_te = Xseq[s_tr], Xseq[s_va], Xseq[s_te]
        ys_tr, ys_va, ys_te = ys[s_tr], ys[s_va], ys[s_te]
        cw_s = self._compute_balanced_weights(ys_tr)

        return DataSplit(Xs_tr, ys_tr, Xs_va, ys_va, Xs_te, ys_te, cw_s), seq_info
