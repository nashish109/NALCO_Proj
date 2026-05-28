from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "datasets"
RESULTS_DIR = ROOT_DIR / "results"

MODEL_PATH = ROOT_DIR / "machine_breakdown_model.pkl"
RESULTS_JSON = ROOT_DIR / "results.json"
PREDICTIONS_PATH = RESULTS_DIR / "machine_failure_predictions.csv"

DEFAULT_TRAIN_DATA = DATASETS_DIR / "machine_vibration_training_dataset.csv"
DEFAULT_PREDICT_DATA = DATASETS_DIR / "machine_vibration_testing_dataset.csv"


def ensure_project_dirs() -> None:
    DATASETS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def resolve_input_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.exists():
        return path

    dataset_path = DATASETS_DIR / path.name
    if dataset_path.exists():
        return dataset_path

    return path
