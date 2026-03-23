from agent.planner import decide_steps

from tools.load_data import load_dataset
from tools.clean_data import clean_data
from tools.summarize import summarize_data
from tools.eda import run_eda
from tools.train_model import train_model

file_path = "data/sample.csv"

user_query = "Analyze this dataset and give insights"

# 🧠 Step 1: LLM decides steps
steps = decide_steps(user_query)

print("\n--- LLM DECIDED STEPS ---")
print(steps)

# 🔧 Step 2: Execute steps
steps_list = steps.lower().split(",")

for step in steps_list:
    step = step.strip()

    if "load" in step:
        print("\n--- LOAD DATA ---")
        print(load_dataset(file_path))

    elif "clean" in step:
        print("\n--- CLEAN DATA ---")
        print(clean_data(file_path))

    elif "eda" in step:
        print("\n--- EDA ---")
        print(run_eda(file_path))

    elif "summarize" in step:
        print("\n--- SUMMARY ---")
        print(summarize_data(file_path))

    elif "train" in step:
        print("\n--- MODEL TRAINING ---")
        print(train_model(file_path))

print("\n--- FINAL DONE ---")