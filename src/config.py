from dataclasses import dataclass, field
from typing import Dict, List, Union


@dataclass
class ExperimentConfig:
    data_path: str = "data"
    label_col: str = "auto"
    out_dir: str = "resultados_poster"
    seed: int = 42
    sample: int = 0
    epochs: int = 30
    optuna_trials: int = 0
    gpu: bool = False
    skip_shap: bool = False
    shap_samples: int = 100
    seq_len: int = 20
    seq_stride: int = 5
    test_size: float = 0.20
    val_size_of_train: float = 0.125
    ece_bins: int = 10
    features: List[str] = field(default_factory=lambda: [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "act_data_pkt_fwd",
        "min_seg_size_forward",
        "Active Mean",
        "Idle Mean",
    ])
    best_mlp_params: Dict[str, Union[int, float]] = field(default_factory=lambda: {
        "n_layers": 2,
        "n_units": 128,
        "dropout_rate": 0.334,
        "lr": 0.000126,
        "batch_size": 512,
    })
    base_mlp_params: Dict[str, Union[int, float]] = field(default_factory=lambda: {
        "n_layers": 2,
        "n_units": 128,
        "dropout_rate": 0.3,
        "lr": 1e-3,
        "batch_size": 512,
    })

