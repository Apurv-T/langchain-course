from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Searches the web for a query and returns the results.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    """Searches the web for a query and returns the results."""
    return tavily_client.search(query=query)


llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(content="3 job postings for ai engineer on linkedin in Toronto"),
        }
    )
    print(result)


if __name__ == "__main__":
    main()
