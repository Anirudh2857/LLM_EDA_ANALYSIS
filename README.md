# Advanced EDA Chatbot with LangChain & LLMs

## Overview
This project is a Streamlit-based chatbot that leverages LangChain and Large Language Models (LLMs) to assist with Exploratory Data Analysis (EDA). Users can upload CSV datasets, filter data, generate visualizations, and interact with an AI-powered chatbot to gain insights.

## Features
- **Data Upload:** Users can upload CSV files via Streamlit's file uploader.
- **Data Filtering:** Provides interactive filtering options for numerical and categorical variables.
- **Basic Data Exploration:** Displays dataset information, summary statistics, and missing values.
- **Visualizations:** Generates correlation heatmaps, pairplots, histograms, boxplots, and categorical distributions.
- **AI-Powered Chatbot:** Users can ask questions about their dataset, and the chatbot will provide insights using pandas, seaborn, and matplotlib.

## Tech Stack
- **Streamlit**: For building the interactive web interface.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **LangChain**: For integrating language models.
- **OpenAI GPT / Local Llama Model**: As the backend AI for answering data-related queries.

## Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install streamlit pandas numpy matplotlib seaborn langchain openai
```

## Usage
1. Run the application:
```bash
streamlit run app.py
```
2. Upload a CSV dataset using the sidebar.
3. Apply filters to explore the data.
4. View dataset summaries and visualizations.
5. Ask questions about the dataset and receive AI-generated insights.

## Deployment
This application is deployed on **Streamlit Cloud**. You can access it via the following link:
https://llmedaanalysis-ehhnbyd8zszndjxvxjbsse.streamlit.app

**Deployment Link:** Please replace `your-streamlit-app-url` with your actual deployment link.

## Configuration
- **OpenAI API Key:** Ensure you have an OpenAI API key stored in the environment variable `OPENAI_API_KEY`.
- **Local Llama Model (Optional):** If using a local model, set `USE_LOCAL_LLM = True` and specify the model path.

## Notes
- The chatbot processes a reduced dataset summary to optimize API calls.
- Some visualizations may take time for large datasets.

## License
This project is open-source and available for modification and improvement.

