import os
import time
import warnings
from typing import Any, Dict

import numpy as np

from src.calibration import ConfidenceCalibrator
from src.config import ExperimentConfig
from src.dataset import DatasetManager
from src.evaluation import MetricsEvaluator
from src.explainability import ExplainabilityAnalyzer
from src.models import (
    CNNLSTMDetector,
    LightGBMDetector,
    LogisticRegressionDetector,
    MLPDetector,
    OptunaMLPTuner,
)
from src.reporting import ReportGenerator

warnings.filterwarnings("ignore")


class DDoSDetectionPipeline:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset_manager = DatasetManager(config)
        self.evaluator = MetricsEvaluator()
        self.explainer = ExplainabilityAnalyzer(config)
        self.calibrator = ConfidenceCalibrator(config.ece_bins)
        self.reporter = ReportGenerator(config.out_dir)
        self.results: Dict[str, Any] = {}

    def _set_seeds(self) -> None:
        import tensorflow as tf

        np.random.seed(self.config.seed)
        tf.random.set_seed(self.config.seed)

    def run(self) -> Dict[str, Any]:
        self._set_seeds()
        t0 = time.time()

        self.reporter.print_section("1. DATASET")
        self.dataset_manager.load()
        tab_split, dataset_summary = self.dataset_manager.prepare_tabular_splits()
        self.results["dataset"] = dataset_summary

        self.reporter.print_section("2. MODELOS TABULARES")
        print("Regresión Logística (línea base)...")
        lr_detector = LogisticRegressionDetector()
        lr_detector.fit(tab_split.X_train, tab_split.y_train)
        self.evaluator.register_model(
            lr_detector.name,
            lr_detector.approach,
            tab_split.y_test,
            lr_detector.predict_proba(tab_split.X_test),
        )

        print("MLP (original)...")
        mlp_base = MLPDetector(
            name="MLP (original)",
            params=self.config.base_mlp_params,
            epochs=self.config.epochs,
        )
        mlp_base.fit(
            tab_split.X_train,
            tab_split.y_train,
            tab_split.X_val,
            tab_split.y_val,
            tab_split.class_weights,
        )
        self.evaluator.register_model(
            mlp_base.name,
            mlp_base.approach,
            tab_split.y_test,
            mlp_base.predict_proba(tab_split.X_test),
        )

        if self.config.optuna_trials > 0:
            print(f"Optuna ({self.config.optuna_trials} trials)...")
            tuner = OptunaMLPTuner(trials=self.config.optuna_trials, seed=self.config.seed)
            best_params = tuner.optimize(
                tab_split.X_train,
                tab_split.y_train,
                tab_split.X_val,
                tab_split.y_val,
                tab_split.class_weights,
            )
        else:
            best_params = dict(self.config.best_mlp_params)
            print("MLP (Optuna): usando hiperparámetros reportados en la tesis:", best_params)

        self.results["mlp_optuna_params"] = best_params
        mlp_optuna = MLPDetector(
            name="MLP (Optuna)",
            params=best_params,
            epochs=self.config.epochs,
        )
        mlp_optuna.fit(
            tab_split.X_train,
            tab_split.y_train,
            tab_split.X_val,
            tab_split.y_val,
            tab_split.class_weights,
        )
        self.evaluator.register_model(
            mlp_optuna.name,
            mlp_optuna.approach,
            tab_split.y_test,
            mlp_optuna.predict_proba(tab_split.X_test),
        )

        print("LightGBM...")
        lgbm_detector = LightGBMDetector(seed=self.config.seed, use_gpu=self.config.gpu)
        lgbm_detector.fit(
            tab_split.X_train,
            tab_split.y_train,
            tab_split.X_val,
            tab_split.y_val,
        )
        p_lgb_test = lgbm_detector.predict_proba(tab_split.X_test)
        p_lgb_val = lgbm_detector.predict_proba(tab_split.X_val)
        self.evaluator.register_model(
            lgbm_detector.name,
            lgbm_detector.approach,
            tab_split.y_test,
            p_lgb_test,
        )

        self.reporter.print_section("3. MATRIZ DE CONFUSIÓN - LightGBM (conjunto de prueba)")
        self.evaluator.print_confusion_matrix("LightGBM", len(tab_split.y_test))

        self.reporter.print_section("4. MODELO SECUENCIAL (CNN-LSTM)")
        seq_split, seq_info = self.dataset_manager.prepare_sequential_splits()
        self.results["secuencias"] = seq_info

        cnn_lstm = CNNLSTMDetector(
            seq_len=self.config.seq_len,
            n_features=len(self.config.features),
            epochs=self.config.epochs,
        )
        cnn_lstm.fit(
            seq_split.X_train,
            seq_split.y_train,
            seq_split.X_val,
            seq_split.y_val,
            seq_split.class_weights,
        )
        p_cnn_test = cnn_lstm.predict_proba(seq_split.X_test)
        p_cnn_val = cnn_lstm.predict_proba(seq_split.X_val)
        self.evaluator.register_model(
            cnn_lstm.name,
            cnn_lstm.approach,
            seq_split.y_test,
            p_cnn_test,
        )

        self.results["modelos"] = self.evaluator.model_metrics

        self.reporter.print_section("5. COMPARATIVA GLOBAL (tabla del póster)")
        table_df = self.evaluator.get_comparison_dataframe()
        print(table_df.round(4).to_string(index=False))
        self.reporter.save_csv(table_df, "tabla_modelos.csv")

        self.reporter.print_section("6. INTERPRETABILIDAD (top 3)")
        perm_top3 = self.explainer.compute_permutation_importance(
            lgbm_detector,
            tab_split.X_test,
            tab_split.y_test,
        )
        self.results["permutation_top3"] = perm_top3

        shap_top3 = self.explainer.compute_kernel_shap(
            cnn_lstm,
            seq_split.X_train,
            seq_split.X_test,
        )
        if shap_top3:
            self.results["shap_top3"] = shap_top3

        self.reporter.save_csv(self.explainer.get_importance_dataframe(), "importancias.csv")

        self.reporter.print_section("7. CALIBRACIÓN DE CONFIANZA (ECE y Brier)")
        self.calibrator.calibrate_lightgbm(
            p_lgb_val,
            tab_split.y_val,
            p_lgb_test,
            tab_split.y_test,
        )
        temperature = self.calibrator.calibrate_cnn_lstm(
            p_cnn_val,
            seq_split.y_val,
            p_cnn_test,
            seq_split.y_test,
        )
        self.reporter.save_csv(self.calibrator.get_calibration_dataframe(), "calibracion.csv")
        self.results["calibracion"] = self.calibrator.calibration_rows
        self.results["temperature"] = temperature

        self.reporter.save_json(self.results, "resultados.json")
        self.reporter.plot_f1_comparison(table_df, "f1_modelos.png")

        elapsed_min = (time.time() - t0) / 60.0
        print(f"\nListo en {elapsed_min:.1f} min. Resultados en: {os.path.abspath(self.config.out_dir)}")
        return self.results
