# Network Intrusion Detection System (NIDS)

A CICIDS2017-based network-flow intrusion detection prototype using a supervised Random Forest classifier and a benign-baseline Isolation Forest anomaly detector, exposed through FastAPI and a browser dashboard.

## 1. Dataset placement

Download the CICIDS2017 daily CSV files and place them in `dataset/`. The preprocessing script expects the standard CICIDS2017 flow feature columns and `Label` column.

## 2. Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3. Rebuild the ML pipeline

Run these commands from the project root, in this order:

```bash
python data_prep/data_prep.py
python feature_engineering/feature_engineering_ml.py
python models/train.py
python models/evaluate.py
```

The scripts create `data/processed/` and `artifacts/` as needed. Preprocessing parameters are fitted only on the training split and saved in `artifacts/preprocessing.pkl`. Random Forest predicts the known attack family; Isolation Forest is trained only on benign training flows. Its decision threshold is selected from benign validation scores.

## 3a. Generate EDA statistics and figures

After data preparation, run:

```bash
python analysis/eda_stats.py
python analysis/eda_visuals.py
python analysis/eda_advanced_visuals.py
```

These commands refresh the dashboard's EDA JSON and plots. The scripts use the shared feature definitions and no longer assume unsupported repeat-call or endpoint-diversity measurements.

## 4. Run the dashboard

From the project root:

```bash
uvicorn dashboard.main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Upload a raw CICIDS2017 CSV containing the expected flow feature columns. `Timestamp` and `Flow ID` are optional metadata columns. The API accepts files up to 100 MB and returns at most 500 flow rows in the table response while computing summary counts over the full upload.

## 5. Important interpretation notes

- Random Forest `confidence` is its maximum class probability and is not guaranteed to be calibrated.
- `anomaly_score` is the negated Isolation Forest decision function; larger values mean more anomalous. It is not a probability.
- `risk_score` is a transparent heuristic indicator: `0.70 × (classifier predicts non-Benign) + 0.30 × (Isolation Forest flags anomaly)`. It is not a calibrated probability of compromise.
- Behaviour tags are descriptive heuristics, not confirmed attack classifications.
- Temporal charts are generated only when a valid timestamp column is present. No synthetic timestamps are created.
- Evaluation on CICIDS2017 is benchmark evaluation, not proof of performance on live networks or unseen real-world attacks.
- The dashboard's EDA tabs use generated JSON/PNG files. Regenerate them after rebuilding the cleaned dataset to avoid viewing stale visual outputs.

## 6. Reproducibility

The project uses a fixed random seed (42). The train/validation/test split is stratified 70/15/15. Preprocessing artifacts, model metadata, and the held-out evaluation report are saved under `artifacts/`.
