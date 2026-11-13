"""Backward-compatible entry point for the single supported NIDS pipeline.

The previous file contained a second, inconsistent training/inference implementation.
Use this orchestrator or run the documented scripts individually.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_ml_pipeline():
    from data_prep.data_prep import main as prepare_data
    from feature_engineering.feature_engineering_ml import main as build_features
    from models.train import main as train_models
    from models.evaluate import main as evaluate_models

    prepare_data()
    build_features()
    train_models()
    evaluate_models()


if __name__ == "__main__":
    run_ml_pipeline()
