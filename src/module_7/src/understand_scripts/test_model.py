import numpy as np
from src.basket_model.basket_model import BasketModel

example_features = np.array([[88.30292, 14.00000, 12.00000, 0.00000]])
model = BasketModel()
prediction = model.predict(example_features)
print("Predicción:", prediction)