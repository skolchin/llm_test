import os
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

model = ChatOllama(model="command-r")

db_file = os.path.abspath('./data/titanic.db')
assert os.path.exists(db_file), f'Database {db_file} does not exist'

db = SQLDatabase.from_uri(f"sqlite:///{db_file}")
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

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
                content=f"Error: {repr(error)}\n please fix your mistakes.",
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
    You are an agent designed to interact with a Titanic database.
    Given an input question, create a syntactically correct SQL query to run, 
    then look at the results of the query and return the answer.

    To start you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step.
    You MUST double check your query before executing it.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    If you get an error while executing a query, rewrite the query and try again.

    You have access to tools for interacting with the Titanic database.
    Only use the below tools. Only use the information returned by the below tools to construct your answer.
    If there's not enought information, use word "Unknown" for that part of answer.

    Use tool 'sql_db_list_tables' to list tables in the Titanic database.
    Use tool 'sql_db_schema' to retrieve structure of a table  in the Titanic database.
    Use tool 'sql_db_query_checker' to double check SQL query before execution.
    Use tool 'sql_db_query' to run SQL query on the Titanic database.
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

def run_and_print(messages):
    for chunk in app.stream(
        {"messages": messages}, 
        stream_mode="values",
        config={"configurable": {"thread_id": 42}}):
            
            chunk["messages"][-1].pretty_print()
            messages.extend(chunk["messages"])

    return messages

messages = run_and_print([
    ("system", system_message),
    ("human", "What tables are available in Titanic database?")
])

messages = run_and_print(messages + [("human", "How many people are in total in Titanic database?")])
messages = run_and_print(messages + [('human', 'How many of them survived?')])
messages = run_and_print(messages + [('human', 'Provide me with a single SQL query to get all the preceding figures')])
