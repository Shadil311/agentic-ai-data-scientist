import streamlit as st
import os
import pandas as pd

from tools.load_data import load_dataset
from tools.clean_data import clean_data
from tools.summarize import summarize_data
from tools.eda import run_eda
from tools.train_model import train_model

# Page config
st.set_page_config(page_title="AI Data Scientist", layout="centered")

# Title
st.title("🤖 Agentic AI Data Scientist")

# Upload file
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# MAIN BLOCK
if uploaded_file is not None:

    # Save file
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully!")

    # Load dataframe
    df = pd.read_csv(file_path)

    # Preview
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Target selection (last column default)
    target_column = st.selectbox(
        "🎯 Select Target Column",
        df.columns,
        index=len(df.columns) - 1
    )

    # ANALYZE BUTTON
    if st.button("🚀 Analyze Dataset"):

        # =========================
        # STEP 1: LOAD DATA
        # =========================
        st.markdown("## 📊 Step 1: Load Data")
        st.success("Dataset loaded successfully")
        st.code(load_dataset(file_path), language="text")
        st.markdown("---")

        # =========================
        # STEP 2: CLEAN DATA
        # =========================
        st.markdown("## 🧹 Step 2: Clean Data")
        st.success("Data cleaned successfully")
        st.code(clean_data(file_path), language="text")
        st.markdown("---")

        # =========================
        # STEP 3: EDA
        # =========================
        st.markdown("## 📈 Step 3: Exploratory Data Analysis")

        run_eda(file_path)
        st.success("EDA completed")

        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists("outputs/histogram.png"):
                st.image("outputs/histogram.png", caption="Histogram")

        with col2:
            if os.path.exists("outputs/correlation.png"):
                st.image("outputs/correlation.png", caption="Correlation Heatmap")

        st.markdown("---")

        # =========================
        # STEP 4: SUMMARY
        # =========================
        st.markdown("## 📋 Step 4: Summary")
        st.code(summarize_data(file_path), language="text")
        st.markdown("---")

        # =========================
        # STEP 5: MODEL TRAINING
        # =========================
        st.markdown("## 🤖 Step 5: Model Training")

        result = train_model(file_path, target_column)

        if "error" in result:
            st.error(result["error"])

        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📌 Problem Type")
                st.success(result["problem_type"].upper())

            with col2:
                st.markdown("### 🎯 Target Column")
                st.info(result["target"])

            st.markdown("### 📊 Model Performance")

            for model, score in result["results"].items():
                st.metric(label=model, value=f"{score:.2f}")

            st.markdown("### 🏆 Best Model")
            st.success(result["best_model"])