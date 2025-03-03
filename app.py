import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.schema import HumanMessage, SystemMessage
import os

# 🔹 Load OpenAI API Key securely
OPENAI_API_KEY = "your-secret-key"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 🔹 Choose between OpenAI (default) or a local Llama model
USE_LOCAL_LLM = False  # Set to True if using a local model

if USE_LOCAL_LLM:
    from langchain.llms import LlamaCpp
    llm = LlamaCpp(model_path="path/to/llama-2-7b.ggmlv3.q4_0.bin")
else:
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.5)

# 🎨 Streamlit UI
st.title("📊 Advanced EDA Chatbot with LangChain & LLMs")
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])

# 🔹 If file uploaded, read and display data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📋 Data Preview")
    st.dataframe(df.head())

    # 🔹 Data Filtering & Interactive Selection
    st.sidebar.subheader("🔍 Data Filtering")
    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    filters = {}
    for col in numerical_cols:
        min_val, max_val = df[col].min(), df[col].max()
        filters[col] = st.sidebar.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
    for col in categorical_cols:
        unique_values = df[col].unique()
        filters[col] = st.sidebar.multiselect(f"Filter {col}", unique_values, default=unique_values)
    
    # Apply filters
    for col, value in filters.items():
        if isinstance(value, tuple):
            df = df[(df[col] >= value[0]) & (df[col] <= value[1])]
        else:
            df = df[df[col].isin(value)]
    
    st.write("### Filtered Data Preview")
    st.dataframe(df.head())

    # 🔹 Basic Dataset Info
    if st.sidebar.checkbox("Show Dataset Info"):
        st.write("### Dataset Information")
        st.write(df.info())

    if st.sidebar.checkbox("Show Summary Statistics"):
        st.write("### Summary Statistics")
        st.write(df.describe())

    if st.sidebar.checkbox("Show Missing Values"):
        st.write("### Missing Values")
        st.write(df.isnull().sum())

    # 🔹 Data Visualizations
    st.sidebar.subheader("📊 Visualization Options")

    # 🔸 Correlation Heatmap
    if st.sidebar.button("Show Correlation Heatmap"):
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col], _ = pd.factorize(df_encoded[col])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)

    # 🔸 Pairplot
    if st.sidebar.checkbox("Show Pairplot"):
        st.write("### 🔗 Pairplot")
        pairplot = sns.pairplot(df)
        st.pyplot(pairplot)

    # 🔸 Histograms
    if st.sidebar.checkbox("Show Histograms"):
        st.write("### 📊 Histograms")
        df.hist(figsize=(12, 8), bins=30)
        st.pyplot(plt)

    # 🔸 Boxplots
    if st.sidebar.checkbox("Show Boxplots"):
        st.write("### 📦 Boxplots")
        for col in numerical_cols:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=col, ax=ax)
            st.pyplot(fig)

    # 🔸 Countplot for Categorical Variables
    if st.sidebar.checkbox("Show Categorical Distributions"):
        st.write("### 📊 Categorical Feature Distributions")
        for col in categorical_cols:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=col, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # 🔹 User query input
    user_query = st.text_input("Ask a question about the dataset:")

    if user_query:
        # 🔹 System prompt to guide the LLM
        system_prompt = "You are an expert data scientist performing EDA. Use pandas, seaborn, and matplotlib for analysis. Provide insights on correlations, trends, and anomalies."

        # 🔹 Reduce dataset summary size to save API tokens
        dataset_summary = df.describe().iloc[:, :3].to_string()

        # 🔹 Construct messages for LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Dataset Summary: {dataset_summary} \n\nUser Query: {user_query}")
        ]

        # 🔹 Get response from LLM
        response = llm(messages)

        # 🔹 Display chatbot response
        st.write("### 🤖 Chatbot Response")
        st.write(response.content)