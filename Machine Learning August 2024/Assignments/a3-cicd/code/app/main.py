from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.api import api_router
from app.api.heartbeat import heartbeat_router
from app.core.config import settings
from app.core.event_handler import start_app_handler, stop_app_handler

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(heartbeat_router)
app.include_router(api_router, prefix=settings.API_V1_STR, tags=["ML API"])

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url=settings.API_V1_STR)

app.add_event_handler("startup", start_app_handler(app, settings.MODEL_PATH, settings.MODEL_PATH2, settings.MODEL_PATH3, settings.X_scaler_path, settings.Y_scaler_path, settings.rf_scaler_path))
app.add_event_handler("shutdown", stop_app_handler(app))

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn
    from app.services.custom import *

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")