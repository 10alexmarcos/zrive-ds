import os
import logging
from joblib import load
from .train_guille import OUTPUT_PATH, feature_label_split
from .utils_guille import build_feature_frame

logger = logging.getLogger(__name__)
logger.level = logging.INFO

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

def main():
    model_name  = "20232006-150917_ridge_1e-06.pkl"
    model = load(os.path.join(OUTPUT_PATH, model_name))
    logger.info("Loaded model {model_name}")

    df = build_feature_frame()
    X, y = feature_label_split(df)

    y_pred = model.predict_proba(X)[:, 1]


if __name__ == "__main__":
    main()