# NIDS Dashboard

The dashboard is the FastAPI + vanilla HTML/CSS/JavaScript interface for the root NIDS project. It now calls the saved Random Forest and Isolation Forest inference pipeline; it does not generate mock predictions.

Follow the root [`README.md`](../README.md) for environment setup, dataset preparation, model training, evaluation, and server startup.

Start the API from the project root:

```bash
uvicorn dashboard.main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Upload a raw CICIDS2017 CSV containing the required flow feature columns. `Timestamp` and `Flow ID` are optional metadata.
