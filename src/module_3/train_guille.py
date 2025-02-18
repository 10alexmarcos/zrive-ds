import pandas as pd
import os
import logging
import joblib
import datetime

from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

from .utils_guille import build_feature_frame, STORAGE_PATH 

logger = logging.getLogger(__name__)
logger.level = logging.INFO

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

HOLDOUT_SIZE = 0.2
RIDGE_Cs = [1e-8, 1e-6, 1e-4, 1e-2]
FEATURE_COLS= [
    "ordered_before",
    "abandoned_before",
    "global_popularity",
    "set_as_Regular"
]
LABEL_COL = "outcome"
OUTPUT_PATH = os.path.join(STORAGE_PATH, "module_3_models")

def evaluate_model(model_name: str, y_test: pd.Series, y_pred: pd.Series) -> float:
    """Evaluate model based on precision-recall AUC. We use ROC AUC as secondary metric"""
    precision_, recall_, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall_, precision_)
    roc_auc = roc_auc_score(y_test, y_pred)
    logger.info(
        f"{model_name} results: {{PR AUC: {pr_auc:.2f}, 'ROC AUC': {roc_auc:.2f}}}"
    )

    return pr_auc

def feature_label_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS], df[LABEL_COL]

def train_test_split(
        df: pd.DataFrame, train_size: float
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """There is a time component to it so we split by date """
    daily_orders = df.groupby("order_date").order_id.nunique()
    cumsum_daily_orders = daily_orders.cumsum() / daily_orders.sum()
    cutoff = cumsum_daily_orders[cumsum_daily_orders <= train_size].idxmax()

    X_train, y_train = feature_label_split(df[df.order_date <= cutoff])
    X_val, y_val = feature_label_split(df[df.order_date > cutoff])
    logger.info(
        "Splitting data on {}. {} training samples and {} validation samples".format(cutoff, X_train.shape[0], X_val.shape[0])
    )

    return X_train, y_train, X_val, y_val


def save_model(model: BaseEstimator, model_name:str) -> None:
    logger.info(f"Saving model {model_name} to {OUTPUT_PATH}")
    if not os.path.exists(OUTPUT_PATH):
        logger.info(f"Creating directory {OUTPUT_PATH}")
        os.makedirs(OUTPUT_PATH)

    model_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_name}.pkl"
    joblib.dump(model, os.path.join(OUTPUT_PATH, model_name))


def ridge_model_selection(df: pd.DataFrame) -> None:
    """
    After exploration we found that some strong regularisation seemed to improve the model.
    However, we prefer to do some selection here with every retrain to make sure that's
    still th eoptimal parameter.
    """
    train_size = 1 - HOLDOUT_SIZE
    X_train, y_train, X_val, y_val = train_test_split(df, train_size=train_size)

    best_auc = 0
    for c in RIDGE_Cs:
        logger.info("Training model with C={c}")
        lr = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", C=c))
        lr.fit(X_train, y_train)

        _ = evaluate_model(
        f"Training with C={c}", y_test=y_train, y_pred=lr.predict_proba(X_train)[:, 1]
        )

        pr_auc = evaluate_model(
            f"Validation with C={c}", y_test=y_val, y_pred=lr.predict_proba(X_val)[:, 1]
        )

        if pr_auc > best_auc:
            logger.info("New best model found")
            best_auc = pr_auc
            best_c = c

    logger.info(f"Training best model with c={best_c} over whole dataset")
    best_model = make_pipeline(
        StandardScaler(), LogisticRegression(penalty="l2", C=best_c)
    )
    X, y =feature_label_split(df)
    best_model.fit(X, y)

    save_model(best_model, f"ridge_{best_c}")

def main():
        feature_frame = build_feature_frame()
        ridge_model_selection(feature_frame)
    
if __name__ == "__main__":
        main()