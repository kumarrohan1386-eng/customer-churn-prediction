FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Python files
COPY main.py .
COPY ui.py .

# Model and data files
COPY customer_churn.csv .
COPY ann_model.pkl .
COPY preprocessor.pkl .

# FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]