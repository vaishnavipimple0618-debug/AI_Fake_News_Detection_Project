# AI-Based Fake News Detection System

A beginner-friendly Streamlit project demonstrating text classification using TF-IDF and Logistic Regression.

## Important limitation
The included training examples are a tiny synthetic demonstration dataset. The output is not a reliable fact-check and should not be used to determine whether real-world news is true. Replace the sample data with a reputable labelled dataset and evaluate the model before making performance claims.

## Run on Windows (VS Code)
1. Install Python 3.10+.
2. Open this folder in VS Code and open Terminal.
3. Optional virtual environment:
   `python -m venv .venv`
   PowerShell: ` .\.venv\Scripts\Activate.ps1`
4. Install packages:
   `python -m pip install -r requirements.txt`
5. Start the app:
   `streamlit run app.py`
6. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Stack
Python, Streamlit, Pandas, scikit-learn, TF-IDF, Logistic Regression.
