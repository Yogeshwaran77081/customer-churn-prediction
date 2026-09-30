"""Central configuration: paths, column names and hyper-parameters."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "Churn_Modelling.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "Exited"
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
CATEGORICAL_ONEHOT = ["Geography"]
BINARY_MAP = {"Gender": {"Female": 0, "Male": 1}}

RANDOM_STATE = 42
TEST_SIZE = 0.20   # held-out test set (real, untouched distribution)
VAL_SIZE = 0.20    # share of the training portion kept as real validation data
THRESHOLD = 0.5

ANN_PARAMS = {"epochs": 100, "batch_size": 100, "patience": 10}
