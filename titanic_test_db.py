# Test LLM with SQL generation

import os
import crewai
from crewai_tools import tool
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    ListSQLDatabaseTool, 
    InfoSQLDatabaseTool, 
    QuerySQLDataBaseTool, 
    QuerySQLCheckerTool
)

load_dotenv()
# set_llm_cache(SQLiteCache(database_path=".langchain.db"))

db = SQLDatabase.from_uri("sqlite:///titanic.db")

# sudo systemctl start ollama
# llm = Ollama(model="llama3.1")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
    huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
    temperature=0.35,
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
    How how many people are in total in Titanic database and how many of them had survived?
'''

## Tools
@tool("tool_tables")
def tool_tables() -> list:
    """ Get all the tables in the database """
    x = ListSQLDatabaseTool(db=db).invoke("")
    return [x]

@tool("tool_schema")
def tool_schema(tables: str | dict | list) -> str:
    """Get table schema """
    tool = InfoSQLDatabaseTool(db=db)
    match tables:
        case str() | list():
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


# SQL
SYS_PROMPT_SQL = ''' Use SQL query to answer {user_input} '''

agent_sql = crewai.Agent(
    role="Data analyst",
    goal=SYS_PROMPT_SQL,
    backstory='''
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct SQL query to run, then look at the results of the query and return the answer.

        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        If the question does not seem related to the database, just return "I don't know" as the answer.

        Use the `tool_tables` to get list of tables in the database.
        Output of this tool is a list of tables in database.
        Only tables returned by the tool can be used to construct SQL queries

        Use the `tool_schema` to get given table fields. 
        Input to this tool is a name of a table, output is the schema and sample rows for this table. 
        Be sure that the table actually exist by calling 'tool_tables' first!.
        Example Input: 'table1'

        Use the `tool_check` to review your queries before executing.
        If the query is not correct, an error message will be returned. 
        If an error is returned, rewrite the query, check the query, and try again. 

        Use the `tool_query` to execute SQL queries. Input to this tool is a correct SQL query, 
        output is a result from the database. If the query is not correct, an error message will be returned. 
        If an error is returned, rewrite the query, check the query, and try again. 
        If you encounter an issue with Unknown column 'xxxx' in 'field list', 
        use tool_schema to find the correct table fields.
     ''',
    tools=[tool_tables, tool_schema, tool_check, tool_query], 
    max_iter=100,
    llm=llm,
    allow_delegation=False, verbose=True)

task_sql = crewai.Task(
    description=SYS_PROMPT_SQL,
    agent=agent_sql,
    expected_output='''Output of the SQL query''')

# Report
SYS_PROMPT_WRITER = '''You write stories based on the work of the data analyst to answer {user_input}'''

agent_writer = crewai.Agent(
    role="Writer",
    goal=SYS_PROMPT_WRITER,
    backstory='''
        You are an experienced writer that writes short novels and stories.
        You use numeric results obtained from database and wrap them up with details to create 
        readable texts.
        Only use the information obtained from database to construct your final answer.
        Do not use any numeric facts which were not provided by the data analyst.

        You emphasise important aspects of a story with markdown tags.
        Example of markdown tags:
            **text** important text
     ''',
    max_iter=100,
    llm=llm,
    allow_delegation=False, verbose=True)

## Task
task_writer = crewai.Task(
    description=SYS_PROMPT_WRITER,
    agent=agent_writer,
    context=[task_sql],
    expected_output='''Text output''')

crew = crewai.Crew(agents=[agent_sql, agent_writer], tasks=[task_sql, task_writer], verbose=False)
res = crew.kickoff(inputs={"user_input": USER_PROMPT})
print(f'Results: {res}')
