from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


def search(query: str):
    """Call to surf the web."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    
    return "It's 90 degrees and sunny."


llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(model=llm, tools=[search])


if __name__ == "__main__":
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    print(response["messages"][-1].content)