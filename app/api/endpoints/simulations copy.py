# ems-backend/app/api/endpoints/simulations.py

import asyncio
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from pathlib import Path

from app.models import pydantic_models as models
from app.api.deps import get_current_user

# --- Pydantic Models for ML Input ---

class VibrationInput(BaseModel):
    features: List[float] = Field(
        ...,
        description="A list of 24 numerical features for vibration diagnosis.",
        min_length=24,
        max_length=24
    )

class SolarInput(BaseModel):
    sequence: List[List[float]]

    @field_validator('sequence')
    def check_sequence_shape(cls, v):
        if len(v) != 24:
            raise ValueError('Sequence must have 24 timesteps.')
        if not all(len(row) == 2 for row in v):
            raise ValueError('Each timestep in the sequence must have 2 features.')
        return v

class MotorFaultInput(BaseModel):
    features: List[float] = Field(
        ...,
        description="A list of 40 numerical features for multi-sensor motor fault diagnosis.",
        min_length=40,
        max_length=40
    )


# --- Load ML Models on Startup ---

router = APIRouter()
models_dir = Path("app/ml-models")
ml_models = {}

try:
    # 1. Load Vibration Diagnosis Model (RandomForest)
    ml_models["vibration_model"] = joblib.load(models_dir / "vibration_model.joblib")
    ml_models["vibration_scaler"] = joblib.load(models_dir / "vibration_scaler.joblib")
    ml_models["vibration_label_encoder"] = joblib.load(models_dir / "vibration_label_encoder.joblib")
    ml_models["vibration_features"] = joblib.load(models_dir / "vibration_model_features.json")

    # 2. Load Solar Forecast Model (LSTM)
    ml_models["solar_model"] = tf.keras.models.load_model(models_dir / "lstm_solar_forecast_model.keras")
    ml_models["solar_scaler"] = joblib.load(models_dir / "lstm_solar_scaler.joblib")
    
    # 3. Load Motor Fault Diagnosis Model (XGBoost)
    ml_models["motor_fault_model"] = xgb.XGBClassifier()
    ml_models["motor_fault_model"].load_model(models_dir / "motor_fault_model.json")
    ml_models["motor_fault_scaler"] = joblib.load(models_dir / "scaler.joblib")
    ml_models["motor_fault_label_encoder"] = joblib.load(models_dir / "label_encoder.joblib")
    
    print("✅ All ML models loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ ML Model loading failed: {e}. Prediction endpoints will not work.")
    ml_models = {}


# --- API Endpoints ---

@router.post("/simulate", response_model=models.SimulationResult)
async def run_simulation(params: models.SimulationParams, current_user: models.User = Depends(get_current_user)):
    await asyncio.sleep(1.0)
    base_cost = 1000 - params.pvCurtail * 10
    base_emissions = 500 - params.pvCurtail * 5
    cost = [base_cost + random.uniform(-50, 50) for _ in range(24)]
    emissions = [base_emissions + random.uniform(-20, 20) for _ in range(24)]
    return {"cost": cost, "emissions": emissions}

@router.post("/predict/vibration", response_model=dict)
async def predict_vibration(input_data: VibrationInput, current_user: models.User = Depends(get_current_user)):
    if not ml_models:
        raise HTTPException(status_code=503, detail="ML models are not available.")
    
    input_df = pd.DataFrame([input_data.features], columns=ml_models["vibration_features"])
    scaled_features = ml_models["vibration_scaler"].transform(input_df)
    prediction_encoded = ml_models["vibration_model"].predict(scaled_features)
    probabilities = ml_models["vibration_model"].predict_proba(scaled_features)
    prediction_decoded = ml_models["vibration_label_encoder"].inverse_transform(prediction_encoded)
    confidence = probabilities[0][prediction_encoded[0]]
    
    return {"prediction": prediction_decoded[0], "confidence": float(confidence)}

@router.post("/predict/solar", response_model=dict)
async def predict_solar(input_data: SolarInput, current_user: models.User = Depends(get_current_user)):
    if not ml_models:
        raise HTTPException(status_code=503, detail="ML models are not available.")

    input_sequence = np.array(input_data.sequence)
    scaled_sequence = ml_models["solar_scaler"].transform(input_sequence)
    
    look_back = 24
    future_predictions_scaled = []
    current_window = list(scaled_sequence)

    for _ in range(96):
        input_for_prediction = np.array([current_window[-look_back:]])
        predicted_power_scaled = ml_models["solar_model"].predict(input_for_prediction, verbose=0)[0][0]
        future_predictions_scaled.append(predicted_power_scaled)
        new_step = [0, predicted_power_scaled]
        current_window.append(new_step)

    dummy_irradiation = np.zeros((len(future_predictions_scaled), 1))
    predictions_to_inverse = np.hstack([dummy_irradiation, np.array(future_predictions_scaled).reshape(-1, 1)])
    inversed_predictions = ml_models["solar_scaler"].inverse_transform(predictions_to_inverse)
    final_forecast = np.maximum(0, inversed_predictions[:, 1]).tolist()

    return {"prediction": final_forecast}

@router.post("/predict/motor-fault", response_model=dict)
async def predict_motor_fault(input_data: MotorFaultInput, current_user: models.User = Depends(get_current_user)):
    if not ml_models:
        raise HTTPException(status_code=503, detail="ML models are not available.")
    
    # Since feature names aren't provided, we create a generic DataFrame
    input_df = pd.DataFrame([input_data.features])
    
    # Preprocess (Scale) the data
    scaled_features = ml_models["motor_fault_scaler"].transform(input_df)
    
    # Predict using the XGBoost model
    prediction_encoded = ml_models["motor_fault_model"].predict(scaled_features)
    probabilities = ml_models["motor_fault_model"].predict_proba(scaled_features)
    
    # Post-process (Decode) the output
    prediction_decoded = ml_models["motor_fault_label_encoder"].inverse_transform(prediction_encoded)
    
    # Get confidence score for the predicted class
    confidence = probabilities[0][prediction_encoded[0]]
    
    return {"prediction": prediction_decoded[0], "confidence": float(confidence)}