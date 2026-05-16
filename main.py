# python3 -m venv .venv
# source .venv/bin/activate
# # Windows: .venv\Scripts\activate
# pip install -U langchain deepagents
# pip install python-dotenv
# pip install langchain-ollama
from langchain.agents import create_agent
from dotenv import load_dotenv
import os



def get_weather(city):
    """Get weather for a given city."""
    return "It's sunny in "+city

agent=create_agent(
    model="ollama:qwen2.5:3b",
    tools=[get_weather],
    system_prompt="""
    你是无所不能的孙悟空，啥都知道，但你绝对不会瞎编乱造
    """,
)
# 本地ollama不需要
# load_dotenv(".env")
# OLLAMA_API_KEY=os.getenv("OLLAMA_API_KEY")
question="What's the weather in 博兴?"

def run():
    result=agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    print(result)
    print(result["messages"][-1].content)

if __name__ == "__main__":
    run()