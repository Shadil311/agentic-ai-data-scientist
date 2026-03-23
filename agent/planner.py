from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

def decide_steps(user_query: str) -> str:
    prompt = f"""
You are an AI Data Scientist.

Available steps:
1. load_data
2. clean_data
3. eda
4. summarize_data
5. train_model

ALWAYS include train_model if the task involves analysis.

Return ONLY step names separated by commas.
Do NOT explain.

User request:
{user_query}
"""

    response = llm.invoke(prompt)
    return response.strip()