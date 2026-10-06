# MLOPS EXP-5 - Iris Flower Prediction API

Based on the provided lab manual: FastAPI REST API with Logistic Regression and Joblib for Iris prediction.

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
uvicorn app.main:app --reload
```

Open Swagger UI: http://127.0.0.1:8000/docs

## Test /predict
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
