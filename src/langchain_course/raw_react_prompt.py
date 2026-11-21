import re
import inspect

from dotenv import load_dotenv
from langsmith import traceable
import ollama
load_dotenv()

#this file is incomplete.
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

tools_for_llm={
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}
def get_tool_description(tool_dict):
    """
    Generate a description of the tools for the LLM.
    """
    tool_descriptions = []
    for tool_name, tool_func in tool_dict.items():
        # __wrapped__ is used to get the original function if it's wrapped by decorators
        original_function=getattr(tool_func, "__wrapped__", tool_func)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or "No description available."
        tool_descriptions.append(f"{tool_name}({signature})\n{docstring}")

    return "\n".join(tool_descriptions)

tool_descriptions = get_tool_description(tools_for_llm)
tool_names = ", ".join(tools_for_llm.keys())
react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""
@traceable(name="react_prompt", run_type="llm")
def ollama_chat_traced(model, messages,options):
   return ollama.chat(
        model=model,
        tools=tools_for_llm, 
        messages=messages,
        options=options
    )

# agent loop ------
# @traceable(name="agent_loop_tool_calling")
# def run_agent(question: str):
#   prompt=react_prompt.format(question=question)
#   scratchpad = ""

#     for iteration in range(MAX_ITERATIONS+1):
#         print(f"\nIteration {iteration + 1}:")
#         full_prompt = prompt + "\n" + scratchpad
#         response = ollama_chat_traced(model=MODEL, 
#                                       messages=[{"role": "user", "content": full_prompt}], 
#                                       options={"stop:": ["Observation:",], "temperature": 0.0})




#         response = llm_bind_tool.invoke( messages)
#         print(f"Agent response: {response.content}")
#         messages.append(response)
#         tool_calls=response.tool_calls
#         if tool_calls!= []:
#             tool_name = tool_calls[0].get("name")
#             tool_args = tool_calls[0].get("args", {})
#             tool_call_id = tool_calls[0].get("id")
#             if tool_name in tools_dict:
#                 tool_result = tools_dict[tool_name].invoke(tool_args)
#                 messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))
#             else:
#                 messages.append(HumanMessage(content=f"Error: Tool '{tool_name}' not found."))
#         else:
#             print("Final answer from agent:", response.content)
#             return response.content
