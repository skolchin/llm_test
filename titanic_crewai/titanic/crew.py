import os
from dotenv import load_dotenv
from functools import cached_property
from langchain_ollama.llms import OllamaLLM
from crewai import Agent, Crew, Process, Task
from langchain_huggingface import HuggingFaceEndpoint
from crewai.project import CrewBase, agent, crew, task, llm
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

load_dotenv()

@CrewBase
class TitanicCrew:
    agents_config = './config/agents.yaml'
    tasks_config = './config/tasks.yaml'

    @cached_property
    def database(self) -> SQLDatabase:
        db_file = os.path.abspath(self.agents_config['data_analyst']['database'])
        assert os.path.exists(db_file), f'Database {db_file} does not exist'

        print(f'Using {db_file} database')
        return SQLDatabase.from_uri(f"sqlite:///{db_file}")

    @llm
    def llama_llm(self):
        return OllamaLLM(model="llama3.1")

    @llm
    def mistral_llm(self):
        return OllamaLLM(model="mistral-nemo", temperature=0.6, top_k=10, top_p=0.5)

    @llm
    def command_r_llm(self):
        return OllamaLLM(model="command-r")
    
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
            memory=True,
            max_iter=100,
            embedder={
                "provider": "ollama",
                "config":{
                    "model": 'mistral-nemo',
                }
            }
        )
