import pandas as pd

def clean_data(file_path: str) -> str:
    try:
        # 🔥 STRONG CLEANING
        file_path = file_path.strip().split("\n")[0]  # remove extra lines
        file_path = file_path.replace("'", "").replace('"', "")

        df = pd.read_csv(file_path)
        df = df.dropna()

        return f"""
Data Cleaned Successfully

New Shape: {df.shape}
"""
    except Exception as e:
        return f"Error cleaning data: {str(e)}"