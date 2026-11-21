from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
model = "gpt-6-luna"


# tools defined
@tool
def get_product_price(product_name: str) -> float:
    """
    Get the price of a product.
    """
    # In a real implementation, this would query a database or an API.
    # Here, we return a mock price for demonstration purposes.
    mock_prices = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
    }
    print(f"Fetching price for product: {product_name}")
    return mock_prices.get(product_name.lower(), 0.0)  # Return 0.0 if product not found


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to a price.
    """
    discount_percentages = {"bronze": 10, "silver": 20, "gold": 30}
    discount_percentage = discount_percentages.get(discount_tier, 0)
    discounted_price = price * (1 - discount_percentage / 100)
    print(f"Applying {discount_percentage}% discount to price: {price}")
    return discounted_price


# agent loop ------
@traceable(name="agent_loop_tool_calling")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(model=model)
    llm_bind_tool = llm.bind_tools(tools)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that can fetch product prices and apply discounts."
                "You have access to the product catalog and a discount tool.\n\n"
                "STRICT RULES:\n"
                "- Always use the available tools to fetch information.\n"
                "- Never make up information or guess values.\n"
                "- If the user does not provide the discount tier, ask for it before applying any discount.\n"
                "You have 2 tools available:\n"
                f"1. {get_product_price.name}: {get_product_price.description}\n"
                f"2. {apply_discount.name}: {apply_discount.description}\n"
                "You can call these tools by using the tool name followed by the required parameters in parentheses.\n"
            )
        ),
        HumanMessage(content=question),
    ]
    for iteration in range(MAX_ITERATIONS+1):
        print(f"\nIteration {iteration + 1}:")
        response = llm_bind_tool.invoke( messages)
        print(f"Agent response: {response.content}")
        messages.append(response)
        tool_calls=response.tool_calls
        if tool_calls!= []:
            tool_name = tool_calls[0].get("name")
            tool_args = tool_calls[0].get("args", {})
            tool_call_id = tool_calls[0].get("id")
            if tool_name in tools_dict:
                tool_result = tools_dict[tool_name].invoke(tool_args)
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))
            else:
                messages.append(HumanMessage(content=f"Error: Tool '{tool_name}' not found."))
        else:
            print("Final answer from agent:", response.content)
            return response.content



if __name__ == "__main__":
    result = run_agent("What is the price of a laptop with a silver discount?")
