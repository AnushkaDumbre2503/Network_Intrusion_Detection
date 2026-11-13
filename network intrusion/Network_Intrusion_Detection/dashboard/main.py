from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4
import logging
import os
import traceback

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="NIDS Dashboard API", description="Network Intrusion Detection System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
results_by_id = {}
latest_result_id = None


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    global latest_result_id
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    temp_path = None
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="CSV exceeds the 100 MB upload limit.")
        with NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as temp:
            temp.write(content)
            temp_path = temp.name

        from models.predict import run_prediction
        result = await run_in_threadpool(run_prediction, temp_path)
        result_id = str(uuid4())
        result["request_id"] = result_id
        results_by_id[result_id] = result
        latest_result_id = result_id
        # Keep only the latest five in-memory results.
        while len(results_by_id) > 5:
            oldest = next(iter(results_by_id))
            del results_by_id[oldest]
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as exc:
        logger.error("Prediction value/runtime error: %s", exc)
        traceback.print_exc()
        return JSONResponse(status_code=422, content={"error": str(exc)})
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {exc}"})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()


@app.get("/results")
async def get_results(request_id: str | None = None):
    selected_id = request_id or latest_result_id
    if not selected_id or selected_id not in results_by_id:
        return JSONResponse(content={
            "summary": {"total_flows": 0, "total_anomalies": 0, "high_risk": 0},
            "table_data": [],
            "temporal_data": {"time_bucket": [], "anomaly_count": [], "available": False},
            "feature_importance": [],
        })
    return JSONResponse(content=results_by_id[selected_id])


app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent / "static"), html=True), name="static")
