from fastapi import FastAPI, HTTPException, Request
import time
import logging
import numpy as np

from src.basket_model.feature_store import FeatureStore
from src.basket_model.basket_model import BasketModel
from src.exceptions import PredictionException, UserNotFoundException

app = FastAPI()
logging.basicConfig(
    filename="service_metrics.txt",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s -  %(message)s'
)

fs = FeatureStore()
model = BasketModel()

@app.get("/status")
async def get_status():
    logging.info("Status endpoint accessed.")
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    start_time = time.time()
    try:
        data = await request.json()
        user_id = data.get("USER_ID")
        if not user_id:
            logging.warning("Missing USER_ID in request")
            raise HTTPException(status_code=400, detail="USER_ID is required")
        

        features = fs.get_features(user_id)
        features_array = np.atleast_2d(features.values)
        
        if features_array.size == 0:
            logging.error(f"User {user_id} has no features")
            raise HTTPException(status_code=422,detail="User exists but has no features")

        
        predictions = model.predict(features_array)
        price = float(predictions.mean())
        
        latency = time.time() - start_time
        logging.info(
            f"Prediction successful for {user_id}: "
            f"price={price:.2f} | "
            f"samples={len(predictions)} | "
            f"latency={latency:.3f}s"
        )
        
        return {
            "price": price
        }

    
    except UserNotFoundException as e:
        logging.error(f"User not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except PredictionException as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error")





