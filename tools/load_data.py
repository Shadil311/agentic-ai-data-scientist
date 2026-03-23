import pandas as pd

def load_dataset(file_path: str) -> str:
    try:
        file_path = file_path.strip().split("\n")[0]
        file_path = file_path.replace("'", "").replace('"', "")

        df = pd.read_csv(file_path)

        return f"""
Dataset Loaded Successfully

Shape: {df.shape}

Columns: {list(df.columns)}

Preview:
{df.head().to_string()}
"""
    except Exception as e:
        return f"Error loading dataset: {str(e)}"