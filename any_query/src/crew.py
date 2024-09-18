import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from functools import cached_property
from langchain_community.llms import Ollama
from crewai import Agent, Crew, Process, Task
from langchain_huggingface import HuggingFaceEndpoint
from urllib.parse import urlparse, urlunparse, parse_qs
from crewai.project import CrewBase, agent, crew, task, llm
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

load_dotenv()

@CrewBase
class AnyQueryCrew:
    agents_config = './config/agents.yaml'
    tasks_config = './config/tasks.yaml'
    
    @cached_property
    def database(self) -> SQLDatabase:
        u = urlparse(self.agents_config['data_analyst']['database'])
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

    @llm
    def llama_llm(self):
        return Ollama(model="llama3.1")

    @llm
    def mistral_llm(self):
        return Ollama(model="mistral-nemo")

    @llm
    def cloud_mistral_llm(self):
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-Nemo-Instruct-2407",
            huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
            temperature=0.35,
        )

    @agent
    def agent_sql(self) -> Agent:
        toolkit = SQLDatabaseToolkit(db=self.database, llm=self.agents_config['data_analyst']['llm'])
        return Agent(
            config=self.agents_config['data_analyst'],
            tools=toolkit.get_tools(),
            verbose=True,
        )
    
    @task
    def task_sql(self) -> Task: 
        return Task(
            config=self.tasks_config['query_task'],
            agent=self.agent_sql(),
        )

    @crew
    def crew(self) -> Crew:
        """ Creates the crew """
        return Crew(
            agents=self.agents,  
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
            cache=True,
            max_iter=100,
            memory=True,
            embedder={
                "provider": "ollama",
                "config":{
                    "model": 'mistral-nemo',
                }
            }
        )
