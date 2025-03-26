from src.basket_model.utils.loaders import load_orders, load_regulars, get_mean_item_price
from src.basket_model.utils.features import build_feature_frame

orders = load_orders()
regulars = load_regulars()
mean_price = get_mean_item_price()
feature_frame = build_feature_frame(orders, regulars, mean_price)
print(feature_frame.head())