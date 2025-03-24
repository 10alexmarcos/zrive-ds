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
    #Endpoint para comprobar el estado del servicio
    #responde con un codigo 200 y un mensaje

    logging.info("Status endpoint accessed.")
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    #endpoint para obtener la predicción del precio de la cesta
    #recibe un JSON con la clave 'USER_ID', obtiene las features y realiza la predicción
    #Registra métricas del servicio y del modelo

    start_time = time.time()
    try:
        data = await request.json()
        user_id = data.get("USER_ID")
        if not user_id:
            raise HTTPException(status_code=400, detail="USER_ID is required")
        
        features = fs.get_features(user_id)

        features_array = np.array(features.values).reshape(1, -1)

        prediction = model.predict(features_array)
        price = prediction[0]

        latency = time.time() - start_time
        logging.info(
            f"Predicition for USER_ID {user_id}: {price} (latency: {latency:.3f} seconds)"
        )

        return {"price": price}
    
    except UserNotFoundException as e:
        logging.error(f"User not found {str(e)}")
        raise HTTPException(status_code=404, detail="User not found in feature store")
    except PredictionException as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during model inference")
    except Exception as e:
        logging.error(f"Unexcpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error")
