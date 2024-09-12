# Test LLM with SQL generation
#
# Based on article
# "Build a Data Scientist AI that can query db with SQL, analyze data with Python, write reports with HTML, and do Machine Learning (No GPU, No APIKEY)"
# by Mauro Di Pietro on TDS

import os
import crewai
from crewai_tools import tool
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import ListSQLDatabaseTool, InfoSQLDatabaseTool, QuerySQLDataBaseTool, QuerySQLCheckerTool

load_dotenv()
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

db = SQLDatabase.from_uri("sqlite:///titanic.db")

# sudo systemctl start ollama
# llm = Ollama(model="llama3.1")
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
    temperature=0.5,
    huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
)

# https://build.nvidia.com/meta/llama-3_1-405b-instruct
# llm = ChatNVIDIA(
#     model='meta/llama-3.1-405b-instruct',
#     api_key=os.environ['NVIDIA_API_KEY'],
#     base_url='https://integrate.api.nvidia.com/v1',
#     temperature=0.2,
#     top_p=0.7,
#     max_tokens=1024,
# )

USER_PROMPT = '''
    How many of people had survived and how many people in total is in Titanic dataset?
'''
SYS_PROMPT = '''
    Use SQL query to answer {user_input}
'''

## Tools
@tool("tool_tables")
def tool_tables() -> str:
    """ Get all the tables in the database """
    x = ListSQLDatabaseTool(db=db).invoke("")
    return [x]

@tool("tool_schema")
def tool_schema(tables: str | dict) -> str:
    """Get table schema """
    tool = InfoSQLDatabaseTool(db=db)
    match tables:
        case str():
            x = tool.invoke(tables)
            return x
        case 'dict':
            keys = [k for k in ['value', 'table_name', 'name'] if k in tables]
            assert keys
            x = tool.invoke(tables[k])
            return x

@tool("tool_query")
def tool_query(sql: str) -> str:
    """ Execute an SQL query """
    return QuerySQLDataBaseTool(db=db).invoke(sql)

@tool("tool_check")
def tool_check(sql: str) -> str:
    """ Check SQL validity """
    return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query":sql})


## Agent
agent_sql = crewai.Agent(
    role="Database Engineer",
    goal=SYS_PROMPT,
    backstory='''
        You are an experienced database engineer that creates and optimize efficient SQL queries.
        Use the `tool_tables` to find tables.
        Use the `tool_schema` to get the metadata for the tables.
        Use the `tool_check` to review your queries before executing.
        Use the `tool_query` to execute SQL queries.
     ''',
    tools=[tool_tables, tool_schema, tool_query, tool_check], 
    max_iter=100,
    llm=llm,
    allow_delegation=False, verbose=True)

## Task
task_sql = crewai.Task(
    description=SYS_PROMPT,
    agent=agent_sql,
    expected_output='''Output of the SQL query''')

crew = crewai.Crew(agents=[agent_sql], tasks=[task_sql], verbose=False)
res = crew.kickoff(inputs={"user_input": USER_PROMPT})
print(f'Results: {res}')
