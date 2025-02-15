import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)
logger.level = logging.INFO
consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

STORAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "zrive-ds/aws")
)

def load_dataset() -> pd.DataFrame:
    dataset_name = "feature_frame.csv"
    loading_file = os.path.join(STORAGE_PATH, dataset_name)
    logger.info(f"Loading dataset from {loading_file}")
    return pd.read_csv(loading_file)


def push_relevant_orders(df: pd.DataFrame, min_products: int=5) -> pd.DataFrame:
    """We are only interested in big enough orders that are profitable"""
    order_size = df.groupby("order_id").outcome.sum()
    orders_of_min_size = order_size[order_size >= min_products].index
    return df.loc[lambda x: x.order_id.isin(orders_of_min_size)]


def build_feature_frame() -> pd.DataFrame:
    logger.info("Building feature frame")
    return (
        load_dataset()
        .pipe(push_relevant_orders)
        .assign(created_at=lambda x: pd.to_datetime(x.created_at))
        .assign(order_date=lambda x: pd.to_datetime[x.order_date].dt.date)
    )