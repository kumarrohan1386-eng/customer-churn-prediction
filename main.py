from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle


# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Customer Churn Prediction using ANN",
    version="1.0"
)


# ==========================================
# Load ANN Model
# ==========================================

with open("ann_model.pkl", "rb") as file:
    model = pickle.load(file)


# ==========================================
# Load StandardScaler
# ==========================================

with open("preprocessor.pkl", "rb") as file:
    scaler = pickle.load(file)


# ==========================================
# Create Training Columns
# ==========================================

# Same dataset used in your notebook
df = pd.read_csv("customer_churn.csv")


# Same preprocessing used in your notebook
df = pd.get_dummies(
    df,
    drop_first=True
)


# Remove target column
X = df.drop("Churn", axis=1)


# Save exact feature names
feature_columns = X.columns.tolist()


print("Number of features:", len(feature_columns))


# ==========================================
# Input Schema
# ==========================================

class CustomerData(BaseModel):

    Names: str

    Age: float

    Total_Purchase: float

    Account_Manager: int

    Years: float

    Num_Sites: float

    Onboard_date: str

    Location: str

    Company: str


# ==========================================
# Home API
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running"
    }


# ==========================================
# Prediction API
# ==========================================

@app.post("/predict")
def predict(data: CustomerData):

    try:

        # ----------------------------------
        # Create input DataFrame
        # ----------------------------------

        input_df = pd.DataFrame([{

            "Names": data.Names,

            "Age": data.Age,

            "Total_Purchase": data.Total_Purchase,

            "Account_Manager": data.Account_Manager,

            "Years": data.Years,

            "Num_Sites": data.Num_Sites,

            "Onboard_date": data.Onboard_date,

            "Location": data.Location,

            "Company": data.Company

        }])


        # ----------------------------------
        # Same preprocessing as notebook
        # ----------------------------------

        input_df = pd.get_dummies(
            input_df,
            drop_first=True
        )


        # ----------------------------------
        # Match training columns
        # ----------------------------------

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # ----------------------------------
        # Scale input
        # ----------------------------------

        input_scaled = scaler.transform(input_df)


        # ----------------------------------
        # ANN Prediction
        # ----------------------------------

        probability = model.predict(
            input_scaled,
            verbose=0
        )[0][0]


        probability = float(probability)


        # ----------------------------------
        # Classification
        # ----------------------------------

        if probability >= 0.5:

            prediction = 1

            result = "Customer is likely to churn"

        else:

            prediction = 0

            result = "Customer is not likely to churn"


        # ----------------------------------
        # Return result
        # ----------------------------------

        return {

            "prediction": prediction,

            "probability": round(probability, 4),

            "result": result

        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )