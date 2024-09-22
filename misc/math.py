import math
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

def add(a: float, b: float) -> float:
    """ Add two numbers and returns the result """
    return float(a) + float(b)

def multiply(a: float, b: float) -> float:
    """ Multiply two numbers and returns the result """
    return float(a) * float(b)

def subtract(a: float, b: float) -> float:
    """ Subtract two numbers and returns the result """
    return float(a) - float(b)

def divide(a: float, b: float) -> float:
    """ Divide two numbers and returns the result """
    return float(a) / float(b)

def floor(a: float) -> int:
    """ Returns an integer part of a number"""
    return int(math.floor(float(a)))

add_tool = FunctionTool.from_defaults(fn=add)
multiply_tool = FunctionTool.from_defaults(fn=multiply)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)
floor_tool = FunctionTool.from_defaults(fn=floor)

llm = Ollama(model="llama3.1")
agent = ReActAgent.from_tools([multiply_tool, add_tool, subtract_tool, divide_tool, floor_tool], llm=llm, verbose=True)

response = agent.chat("What is integer part of 3.2 + (5.6 * 2) - 1.3?")
print(response)
