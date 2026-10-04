import streamlit as st
import requests


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# ==========================================
# Title
# ==========================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer details below to predict whether "
    "the customer is likely to churn."
)


# ==========================================
# Customer Input
# ==========================================

Names = st.text_input(
    "Customer Name",
    placeholder="e.g. Cameron Williams"
)

Age = st.number_input(
    "Age",
    min_value=1.0,
    max_value=100.0,
    value=42.0
)

Total_Purchase = st.number_input(
    "Total Purchase",
    min_value=0.0,
    value=11066.80
)

Account_Manager = st.selectbox(
    "Account Manager",
    [0, 1]
)

Years = st.number_input(
    "Years",
    min_value=0.0,
    max_value=100.0,
    value=7.22
)

Num_Sites = st.number_input(
    "Number of Sites",
    min_value=0.0,
    max_value=100.0,
    value=8.0
)

Onboard_date = st.text_input(
    "Onboard Date",
    value="2013-08-30 07:00:40"
)

Location = st.text_input(
    "Location",
    placeholder="Enter customer location"
)

Company = st.text_input(
    "Company",
    placeholder="Enter company name"
)


# ==========================================
# Prediction Button
# ==========================================

if st.button("🔮 Predict Churn"):

    # Check required fields
    if Names == "" or Location == "" or Company == "":

        st.warning("⚠️ Please fill all required fields.")

    else:

        # ==========================================
        # Data for FastAPI
        # ==========================================

        data = {
            "Names": Names,
            "Age": Age,
            "Total_Purchase": Total_Purchase,
            "Account_Manager": Account_Manager,
            "Years": Years,
            "Num_Sites": Num_Sites,
            "Onboard_date": Onboard_date,
            "Location": Location,
            "Company": Company
        }


        try:

            # ==========================================
            # Send Data to FastAPI
            # ==========================================

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=data
            )


            # ==========================================
            # API Response
            # ==========================================

            if response.status_code == 200:

                result = response.json()

                prediction = result["prediction"]
                probability = result["probability"]
                message = result["result"]


                # ==========================================
                # Display Result
                # ==========================================

                st.subheader("Prediction Result")


                if prediction == 1:

                    st.error(
                        "⚠️ Customer is likely to churn"
                    )

                else:

                    st.success(
                        "✅ Customer is not likely to churn"
                    )


                st.write(
                    f"**Probability:** {probability:.2%}"
                )

                st.info(message)


            else:

                st.error(
                    f"❌ API Error: {response.status_code}"
                )

                # Show FastAPI error
                try:
                    error = response.json()
                    st.code(error)
                except:
                    st.write(response.text)


        except requests.exceptions.ConnectionError:

            st.error(
                "❌ FastAPI server is not running."
            )

            st.info(
                "First run FastAPI using: "
                "`uvicorn main:app --reload`"
            )


        except Exception as e:

            st.error(
                f"❌ Error: {str(e)}"
            )