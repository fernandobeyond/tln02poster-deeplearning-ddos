import argparse

from src.config import ExperimentConfig


class CLIApplication:
    def __init__(self):
        self.parser = self._build_parser()

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        ap = argparse.ArgumentParser(
            description="Sistemas de Detección de Ataques DDoS basados en Deep Learning en Entornos de Red"
        )
        ap.add_argument("--data", default="data", help="CSV o carpeta con archivos CSV (por defecto: data)")
        ap.add_argument("--label-col", default="auto", help="Columna de etiqueta (por defecto: auto)")
        ap.add_argument("--out", default="resultados_poster", help="Directorio de salida")
        ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
        ap.add_argument("--sample", type=int, default=0, help="Usar solo N filas (prueba rápida)")
        ap.add_argument("--epochs", type=int, default=30, help="Épocas de entrenamiento")
        ap.add_argument(
            "--optuna-trials",
            type=int,
            default=0,
            help="0 = usar los hiperparámetros reportados en la tesis",
        )
        ap.add_argument("--gpu", action="store_true", help="LightGBM con device='gpu'")
        ap.add_argument("--skip-shap", action="store_true", help="Omitir cálculo de KernelSHAP")
        ap.add_argument("--shap-samples", type=int, default=100, help="Muestras de prueba para KernelSHAP")
        return ap

    def parse_config(self) -> ExperimentConfig:
        args = self.parser.parse_args()
        return ExperimentConfig(
            data_path=args.data,
            label_col=args.label_col,
            out_dir=args.out,
            seed=args.seed,
            sample=args.sample,
            epochs=args.epochs,
            optuna_trials=args.optuna_trials,
            gpu=args.gpu,
            skip_shap=args.skip_shap,
            shap_samples=args.shap_samples,
        )

    def execute(self) -> None:
        from src.pipeline import DDoSDetectionPipeline

        config = self.parse_config()
        pipeline = DDoSDetectionPipeline(config)
        pipeline.run()


def main() -> None:
    app = CLIApplication()
    app.execute()


if __name__ == "__main__":
    main()
