# sudo systemctl start ollama
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from urllib.parse import urlparse, urlunparse, parse_qs
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from typing import Any

load_dotenv()

def get_database(uri) -> SQLDatabase:
    u = urlparse(uri)
    qs = parse_qs(u.query)

    db_url = urlunparse(u._replace(query=None)) \
                .format(
                    POSTGRES_USER=os.environ['POSTGRES_USER'],
                    POSTGRES_PASSWORD=os.environ['POSTGRES_PASSWORD'],
                )

    schema = qs.get('schema')
    if isinstance(schema, (tuple, list)):
        schema = schema[0]

    print(f'Connecting to {db_url}/{schema}')
    return SQLDatabase(create_engine(db_url), schema)

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

model = ChatOllama(model="command-r")
# model = ChatHuggingFace(
#     llm= HuggingFaceEndpoint(
#         repo_id="CohereForAI/c4ai-command-r-08-2024",
#         huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
#     )
# )

db = get_database(os.environ['POSTGRES_URL'])
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
tool_map = {t.name: t for t in tools}

def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

def should_use_tools(state: MessagesState) -> str:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"

    return END

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {error}, please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# output = tools_node.invoke({'messages': [
#     AIMessage(
#         content='',
#         tool_calls=[{
#             "name": "sql_db_list_tables",
#             "args": {},
#             "id": "tool_call_id",
#             "type": "tool_call",
#         }],
#     )
# ]})
# print(output)

system_message = """
    You are an agent designed to interact with a database.
    Given an input question, create a syntactically correct SQL query to run, 
    then look at the results of the query and return the answer.

    To start you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step.
    You must analyze table structure before making a query. Do NOT skip this step.
    You must double check your query before executing it. Do NOT skip this step.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    If you get an error while executing a query, rewrite the query and try again.

    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your answer.
    If there's not enought information, use word "Unknown" for that part of answer.

    Use tool 'sql_db_list_tables' to list tables in the database. The tool takes no input and 
    returns names of tables in a database separated by comma.

    Use tool 'sql_db_schema' to analyze structure of a table  in the database. Input must be 
    a table name as an input and returns table structure along with sample rows.

    Use tool 'sql_db_query_checker' to double check SQL query before execution. Input must be a 
    a SQL query and returns rewritten SQL query or error. If an error was returned, rewrite the query
    and try again.

    Use tool 'sql_db_query' to run SQL query on the database. Input must be a SQL query
    and returns results of its execution or error. If an error was returned, rewrite the query, double check it
    and try again.
"""

model = model.bind_tools(tools)

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", 
                  ToolNode(tools).with_fallbacks(
                       [RunnableLambda(handle_tool_error)], exception_key="error"))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges('agent', should_use_tools)
workflow.add_edge("tools", 'agent')
app = workflow.compile(checkpointer=MemorySaver(), debug=False)

# with open('lng.png', 'wb') as fp:
#     fp.write(app.get_graph().draw_mermaid_png())

QUESTION = "How many records in 'transl_budget' table belongs to the last budget version?"

def run_and_print(messages):
    for chunk in app.stream(
        {"messages": messages}, 
        stream_mode="values",
        config={"configurable": {"thread_id": 42}}):
            
            chunk["messages"][-1].pretty_print()
            messages.extend(chunk["messages"])

    return messages

messages = run_and_print([
    ("human", QUESTION)
])
