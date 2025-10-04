# ems-backend/app/api/endpoints/actions.py

import asyncio
from fastapi import APIRouter, Depends, HTTPException
from app.models import pydantic_models as models
from app.data.mock_data import MOCK_ALERTS, MOCK_RL_SUGGESTIONS
from app.api.deps import get_current_user

router = APIRouter()

@router.post("/sites/{site_id}/alerts/{alert_id}/acknowledge", response_model=dict)
async def acknowledge_alert(site_id: str, alert_id: str, current_user: models.User = Depends(get_current_user)):
    await asyncio.sleep(0.5)
    if site_id in MOCK_ALERTS:
        for alert in MOCK_ALERTS[site_id]:
            if alert.id == alert_id:
                alert.status = 'acknowledged'
                return {"success": True}
    raise HTTPException(status_code=404, detail="Alert not found")

@router.post("/sites/{site_id}/suggestions/{suggestion_id}/accept", response_model=dict)
async def accept_suggestion(site_id: str, suggestion_id: str, current_user: models.User = Depends(get_current_user)):
    await asyncio.sleep(0.8)
    if site_id in MOCK_RL_SUGGESTIONS:
        for suggestion in MOCK_RL_SUGGESTIONS[site_id]:
            if suggestion.id == suggestion_id:
                suggestion.status = 'accepted'
                return {"success": True, "schedule": "Action scheduled for next control cycle."}
    raise HTTPException(status_code=404, detail="Suggestion not found")

@router.post("/sites/{site_id}/suggestions/{suggestion_id}/reject", response_model=dict)
async def reject_suggestion(site_id: str, suggestion_id: str, current_user: models.User = Depends(get_current_user)):
    await asyncio.sleep(0.8)
    if site_id in MOCK_RL_SUGGESTIONS:
        for suggestion in MOCK_RL_SUGGESTIONS[site_id]:
            if suggestion.id == suggestion_id:
                suggestion.status = 'rejected'
                return {"success": True}
    raise HTTPException(status_code=404, detail="Suggestion not found")

@router.post("/sites/{site_id}/maintenance/{asset_id}/schedule", response_model=dict)
async def schedule_maintenance(site_id: str, asset_id: str, current_user: models.User = Depends(get_current_user)):
    await asyncio.sleep(1.2)
    return {"success": True, "message": f"Maintenance for asset {asset_id} has been scheduled."}

@router.post("/sites/{site_id}/rl-strategy", response_model=dict)
async def update_rl_strategy(site_id: str, strategy: models.RLStrategy, current_user: models.User = Depends(get_current_user)):
    await asyncio.sleep(1)
    print(f"Site {site_id} RL strategy updated to: {strategy.dict()}")
    return {"success": True}

@router.post("/alerts/analyze-root-cause", response_model=str)
async def analyze_root_cause(alert: models.Alert, current_user: models.User = Depends(get_current_user)):
    await asyncio.sleep(2.5) # Simulate heavy AI processing
    response = f"""
### Root Cause Analysis for Alert: `{alert.id}`

**Alert Message:** {alert.message}
**Device:** {alert.device_id}

---

**1. Primary Causal Factor:**
Based on historical data for this asset type, the most probable cause is **{alert.diagnosis}**. The system cross-referenced 1,532 similar events and found a 82% correlation with this diagnosis.

**2. Contributing Factors:**
* **Operating Conditions:** The device has been operating at 95% capacity for the last 72 hours, which is above the recommended average of 80%.
* **Maintenance History:** The last scheduled maintenance was postponed by 2 weeks. This could have exacerbated underlying wear and tear.

**3. Recommended Action Confidence:**
The confidence score for the recommended action, "{alert.recommended_action}", is **95%**. This action has resolved similar issues in 9 out of 10 past instances.
"""
    return response