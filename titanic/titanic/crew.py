import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
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
    
    @llm
    def llama_llm(self):
        return Ollama(model="llama3.1")

    @llm
    def mistral_llm(self):
        return Ollama(model="mistral-nemo")

    @llm
    def hf_mistral_llm(self):
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-Nemo-Instruct-2407",
            huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
            temperature=0.35,
        )

    @agent
    def agent_sql(self) -> Agent:
        db_file = os.path.abspath(self.agents_config['data_analyst']['database'])
        assert os.path.exists(db_file), f'Database {db_file} does not exist'

        print(f'Using {db_file} database')
        db = SQLDatabase.from_uri(f"sqlite:///{db_file}")
        toolkit = SQLDatabaseToolkit(db=db, llm=self.agents_config['data_analyst']['llm'])

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
        """ Creates the crea """
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
