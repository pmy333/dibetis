import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="ML Model App",
    page_icon="🤖",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# -------------------------------
# Title & Description
# -------------------------------
st.title("🤖 Machine Learning Prediction App")
st.markdown("### Enter input features to get predictions")

st.write("---")

# -------------------------------
# Input Section (Dynamic Style)
# -------------------------------
st.subheader("🔢 Input Features")

# 👉 EDIT THESE based on your model
col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)

with col2:
    feature3 = st.number_input("Feature 3", value=0.0)
    feature4 = st.number_input("Feature 4", value=0.0)

# Convert input to array
input_data = np.array([[feature1, feature2, feature3, feature4]])

st.write("---")

# -------------------------------
# Prediction Section
# -------------------------------
if st.button("🚀 Predict"):
    try:
        prediction = model.predict(input_data)

        st.success("✅ Prediction Successful!")
        st.metric(label="Prediction Result", value=prediction[0])

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -------------------------------
# Batch Prediction (CSV Upload)
# -------------------------------
st.write("---")
st.subheader("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        preds = model.predict(data)

        data["Prediction"] = preds

        st.write("### 📊 Results")
        st.dataframe(data)

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Results",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.caption("Built with ❤️ using Streamlit")
