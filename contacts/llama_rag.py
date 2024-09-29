# sudo systemctl start ollama
import os
import xml.etree.ElementTree as ET
from functools import cache
from pydantic import Field
from textwrap import dedent
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import XMLReader
from llama_index.core.agent import ReActAgent
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.types import ChatMessage, MessageRole
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata

from rag_tools import *

load_dotenv()

@cache
def get_data() -> ET.Element:
    """ Returns XML for contact list """
    with open('data/contacts.xml', 'rt') as fp:
        return ET.fromstring(fp.read())

@cache
def get_data_lower() -> ET.Element:
    """ Returns XML for contact list with all texts in lower case """
    with open('data/contacts.xml', 'rt') as fp:
        return ET.fromstring(fp.read().lower())

def find_person(name: str = Field('Person name')) -> str:
    """ Use this tool to find a person by name.

        Input to this tool must be full person name.

        The tool returns person ID or multiple IDs separated by comma.
        If no such person was found, 'Not found' will be returned.

    """
    return find_person_impl(get_data_lower(), name)

def get_person_details(person_id: str | int | list = Field('Person ID')) -> str:
    """ Use this tool to retrieve all details on single or multiple person.

        Input to this tool must be either a single person ID or
        multiple person IDs separated by comma. 

        The tool returns a list of person details separated by line feed,
        with each line containing person details separated by comma.
    
    """
    return get_person_details_impl(get_data(), person_id)

def find_departments(dept: str = Field('Department name and office location separated by comma')) -> str:
    """ Use this tool to get a list of departments IDs with given name
        and office location, if applicable. 

        Input to this tool must be either a single department name
        or a department and office location name separated by comma.

        The tool returns department ID or multiple IDs separated by comma.
    """
    return find_departments_impl(get_data_lower(), dept)

def get_department_staff(dept_id: str | int | list = Field('Department ID')) -> str:
    """ Use this tool to get a list of person IDs who works for a given department 
        or multiple departments.

        Input to this tool must be either a single department ID or
        multiple department IDs separated by comma.

        The tool returns list of person IDs separated by comma.
    """
    return get_department_staff_impl(get_data(), dept_id)

llm = Ollama(model="llama3.1", request_timeout=1000.0)
emded_llm = OllamaEmbedding(model_name="llama3.1")
query_llm = Ollama(model="llama3.1", 
                   request_timeout=1000.0, 
                   system_prompt=SYS_PROMPT_REACT_QUERY)

# llm = HuggingFaceInferenceAPI(
#     model_name='mistralai/Mistral-Nemo-Instruct-2407',
#     token=os.environ['HUGGINGFACE_API_KEY'],
#     task='generation',
#     temperature=0.35,
#     top_p=0.8,
# )
# emded_llm = HuggingFaceEmbedding(model_name='BAAI/bge-m3', device='cuda')

Settings.llm = llm
Settings.embed_model = emded_llm

documents = SimpleDirectoryReader('./data/', 
                                  required_exts=['.xml'], 
                                  file_extractor={'.xml': XMLReader()}).load_data()
vector_index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(vector_index, verbose=True)

tools = [
    FunctionTool.from_defaults(find_person),
    FunctionTool.from_defaults(get_person_details),
    FunctionTool.from_defaults(find_departments),
    FunctionTool.from_defaults(get_department_staff),
    QueryEngineTool(
        query_engine=RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=query_llm,
            verbose=True),
        metadata=ToolMetadata(
            name="search_contact_list",
            description="Contact list search tool",
        ),
    ),
]

agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=30,
    context=SYS_PROMPT_REACT,
)
result = agent.chat('Who is responsible for breakfast in Moscow?')
print(result)
