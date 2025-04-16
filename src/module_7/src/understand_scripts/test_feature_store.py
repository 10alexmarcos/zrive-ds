from src.basket_model.feature_store import FeatureStore

fs = FeatureStore()
try:
    user_features = fs.get_features("329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700573ca76f11b45151d67944f171a88fd4f860f06d662c7b29d7b91f0dbc8bf14d410a169a0ed531040b")
    print("features user\n", user_features)
except Exception as e:
    print("Error:", e)
