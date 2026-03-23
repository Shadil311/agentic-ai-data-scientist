from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from tools.summarize import summarize_data

from tools.load_data import load_dataset
from tools.clean_data import clean_data

tools = [
    Tool(
        name="Load Dataset",
        func=load_dataset,
        description="Load dataset using file path like 'data/sample.csv'"
    ),
    Tool(
        name="Clean Data",
        func=clean_data,
        description="Clean dataset using file path like 'data/sample.csv'"
    ),
    Tool(
        name="Summarize Data",
        func=summarize_data,
        description="Summarize dataset and show statistics. Input must be file path like 'data/sample.csv'"
    )
]

# Use local Ollama model
llm = Ollama(model="llama3")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iterations=10,
    early_stopping_method="generate"
)