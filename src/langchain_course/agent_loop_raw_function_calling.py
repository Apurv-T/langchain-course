from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
model = "gpt-6-luna"


# tools defined
@traceable(name="get_product_price")
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


@traceable(name="apply_discount")
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to a price.
    """
    discount_percentages = {"bronze": 10, "silver": 20, "gold": 30}
    discount_percentage = discount_percentages.get(discount_tier, 0)
    discounted_price = price * (1 - discount_percentage / 100)
    print(f"Applying {discount_percentage}% discount to price: {price}")
    return discounted_price

#Ollama 
tools_for_llm=[
{
    "type": "function",
    "function": {
        "name": "get_product_price",
        "description": "Get the price of a product.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "The name of the product. e.g., 'laptop', 'smartphone', 'headphones'."
                }
            },
            "required": ["product_name"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "apply_discount",
        "description": "Apply a discount to a price.",
        "parameters": {
            "type": "object",
            "properties": {
                "price": {
                    "type": "number",
                    "description": "The original price."
                },
                "discount_tier": {
                    "type": "string",
                    "description": "The discount tier. e.g., 'bronze', 'silver', 'gold'."
                }
            },
            "required": ["price", "discount_tier"]
        }
    }
}
]

@traceable(name="Ollama chat", run_type="llm")
def run_ollama_chat(messages):
    return ollama.chat(
        model=model,
        tools=tools_for_llm,
        messages=messages
    )

@traceable(name="agent_loop_tool_calling")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that can fetch product prices and apply discounts."
                "You have access to the product catalog and a discount tool.\n\n"
                "STRICT RULES:\n"
                "- Always use the available tools to fetch information.\n"
                "- Never make up information or guess values.\n"
                "- If the user does not provide the discount tier, ask for it before applying any discount.\n"
                "You have 2 tools available:\n"
                f"1. get_product_price: Get the price of a product.\n"
                f"2. apply_discount: Apply a discount to a price.\n"
                "You can call these tools by using the tool name followed by the required parameters in parentheses.\n"
            )
        },
        {
            "role": "user",
            "content": question
        }
        
    ]
    for iteration in range(MAX_ITERATIONS):
        print(f"\nIteration {iteration + 1}:")
        response = run_ollama_chat(messages)
        print(f"Agent response: {response}")
        messages.append(response)
        tool_calls = response.message.get("tool_calls", [])
        if tool_calls:
            tool_name = tool_calls[0].get("name")
            tool_args = tool_calls[0].get("args", {})
            if tool_name in tools_dict:
                tool_result = tools_dict[tool_name](tool_args)
                messages.append({
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_calls[0].get("id")
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"Error: Tool '{tool_name}' not found."
                })
        else:
            print("Final answer from agent:", response.message.get("content"))
            return response.message.get("content")  

   


# # agent loop ------
# @traceable(name="agent_loop_tool_calling")
# def run_agent(question: str):
#     tools = [get_product_price, apply_discount]
#     tools_dict = {tool.name: tool for tool in tools}
#     llm = init_chat_model(model=model)
#     llm_bind_tool = llm.bind_tools(tools)
#     messages = [
#         SystemMessage(
#             content=(
#                 "You are a helpful assistant that can fetch product prices and apply discounts."
#                 "You have access to the product catalog and a discount tool.\n\n"
#                 "STRICT RULES:\n"
#                 "- Always use the available tools to fetch information.\n"
#                 "- Never make up information or guess values.\n"
#                 "- If the user does not provide the discount tier, ask for it before applying any discount.\n"
#                 "You have 2 tools available:\n"
#                 f"1. {get_product_price.name}: {get_product_price.description}\n"
#                 f"2. {apply_discount.name}: {apply_discount.description}\n"
#                 "You can call these tools by using the tool name followed by the required parameters in parentheses.\n"
#             )
#         ),
#         HumanMessage(content=question),
#     ]
#     for iteration in range(MAX_ITERATIONS + 1):
#         print(f"\nIteration {iteration + 1}:")
#         response = llm_bind_tool.invoke(messages)
#         print(f"Agent response: {response.content}")
#         messages.append(response)
#         tool_calls = response.tool_calls
#         if tool_calls != []:
#             tool_name = tool_calls[0].get("name")
#             tool_args = tool_calls[0].get("args", {})
#             tool_call_id = tool_calls[0].get("id")
#             if tool_name in tools_dict:
#                 tool_result = tools_dict[tool_name].invoke(tool_args)
#                 messages.append(
#                     ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
#                 )
#             else:
#                 messages.append(
#                     HumanMessage(content=f"Error: Tool '{tool_name}' not found.")
#                 )
#         else:
#             print("Final answer from agent:", response.content)
#             return response.content


if __name__ == "__main__":
    result = run_agent("What is the price of a laptop with a silver discount?")
