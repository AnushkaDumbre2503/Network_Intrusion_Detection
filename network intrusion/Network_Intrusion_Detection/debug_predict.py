"""Quick debug script to test prediction pipeline and surface the real error."""
import sys
import traceback

try:
    from models.predict import run_prediction
    # Try using the test_data.csv as a proxy to see if pipeline works at all
    result = run_prediction("data/processed/test_data.csv")
    print("SUCCESS! Keys:", list(result.keys()))
    print("Summary:", result["summary"])
except Exception as exc:
    print(f"ERROR TYPE: {type(exc).__name__}")
    print(f"ERROR MESSAGE: {exc}")
    traceback.print_exc()
