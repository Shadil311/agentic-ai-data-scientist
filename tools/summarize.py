import pandas as pd

def summarize_data(file_path: str) -> str:
    try:
        file_path = file_path.strip().split("\n")[0]
        file_path = file_path.replace("'", "").replace('"', "")

        df = pd.read_csv(file_path)

        return f"""
Dataset Summary

Shape: {df.shape}

Columns: {list(df.columns)}

Stats:
{df.describe().to_string()}
"""
    except Exception as e:
        return f"Error summarizing data: {str(e)}"