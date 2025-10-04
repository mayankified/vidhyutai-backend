# ems-backend/app/api/endpoints/simulations.py

import asyncio
import random
from typing import List
from fastapi import APIRouter, Depends
from app.models import pydantic_models as models
from app.api.deps import get_current_user

router = APIRouter()

@router.post("/simulate", response_model=models.SimulationResult)
async def run_simulation(params: models.SimulationParams, current_user: models.User = Depends(get_current_user)):
    await asyncio.sleep(2.0)
    # Simulate a calculation based on params
    base_cost = 1000 - params.pvCurtail * 10
    base_emissions = 500 - params.pvCurtail * 5
    
    cost = [base_cost + random.uniform(-50, 50) for _ in range(24)]
    emissions = [base_emissions + random.uniform(-20, 20) for _ in range(24)]
    
    return {"cost": cost, "emissions": emissions}

@router.post("/predict/vibration", response_model=dict)
async def predict_vibration(current_user: models.User = Depends(get_current_user)):
    await asyncio.sleep(0.9)
    pred = random.choice(["Nominal", "Bearing Wear Detected", "Misalignment Fault"])
    conf = round(random.uniform(0.75, 0.98), 2)
    return {"prediction": pred, "confidence": conf}

@router.post("/predict/solar", response_model=dict)
async def predict_solar(current_user: models.User = Depends(get_current_user)):
    await asyncio.sleep(1.5)
    # Simulate a 24-hour forecast
    forecast = [max(0, 1000 * (-(i - 12)**2 / 30 + 1) + random.uniform(-50, 50)) for i in range(24)]
    return {"prediction": forecast}

@router.post("/predict/motor-fault", response_model=dict)
async def predict_motor_fault(current_user: models.User = Depends(get_current_user)):
    await asyncio.sleep(1.2)
    pred = random.choice(["No Fault Expected", "Stator Winding Fault Imminent"])
    conf = round(random.uniform(0.88, 0.99), 2)
    return {"prediction": pred, "confidence": conf}