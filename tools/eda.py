import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(file_path: str) -> str:
    try:
        file_path = file_path.strip().split("\n")[0]
        file_path = file_path.replace("'", "").replace('"', "")

        df = pd.read_csv(file_path)

        # 🔥 create folder if not exists
        os.makedirs("outputs", exist_ok=True)

        # Histogram
        df.hist(figsize=(8,6))
        plt.tight_layout()
        plt.savefig("outputs/histogram.png")
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(6,4))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.savefig("outputs/correlation.png")
        plt.close()

        return "EDA completed. Check outputs folder."

    except Exception as e:
        return f"Error in EDA: {str(e)}"