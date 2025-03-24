from src.basket_model.feature_store import FeatureStore
from src.basket_model.basket_model import BasketModel

fs = FeatureStore()
model = BasketModel()

user_id = "500027bf392bfa9ef527919569fba44904d429155b7cf46a992dbea492cc9f0a372d61d3daf8464bed9713ee8e79e04de9d9590d11d2bfe3796ec3f5a4cac625"

try:
    features = fs.get_features(user_id)
    features_array = features.values.reshape(1, -1)
    prediction = model.predict(features_array)
    print("Predicción:", prediction)
except Exception as e:
    print("Error:", e)
