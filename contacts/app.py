# sudo systemctl start ollama
# streamlit run app.py

import json
import streamlit as st
from pydantic import Field
from llama_index.core import Settings
from streamlit_tree_select import tree_select
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.readers.file import XMLReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.types import ChatMessage, MessageRole
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata

from get_model import *
from rag_tools import *

# Defaults for UI elements
MODEL_PARAMS = {
    'mode': 'LLM',
    'model':'llama3.1',
    'embed':'llama3.1',
    'temperature': 0.3,
    'top_p': 0.9,
}

# Data loading
@st.cache_data
def load_labels() -> dict:
    with open('./../data/labels.json', 'rt') as fp:
        nodes = json.load(fp)
    return nodes

@st.cache_data
def get_data() -> ET.Element:
    with open('./../data/contacts.xml', 'rt') as fp:
        return ET.fromstring(fp.read())

@st.cache_data
def get_data_lower() -> ET.Element:
    with open('./../data/contacts.xml', 'rt') as fp:
        return ET.fromstring(fp.read().lower())

# Tools (LI wrappers on functions with additional metadata)
def find_person(name: str = Field('Person name')) -> str:
    """ Use this tool to find a person by name.

        Input to this tool must be a full person name.

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


# Model loading
@st.cache_resource
def make_chat(mode: str, model: str, embed: str, temperature: float, top_p: float):

    # Get the models
    llm, embed_llm = get_model(model, embed, temperature=temperature, top_p=top_p)
    Settings.llm = llm
    Settings.embed_model = embed_llm

    # Load documents to in-memory index
    documents = SimpleDirectoryReader('./../data/', 
                                      required_exts=['.xml'],
                                      file_extractor={'.xml': XMLReader()}).load_data()
    vector_index = VectorStoreIndex.from_documents(documents)

    # Build a chat engine
    match mode:
        case 'LLM':
            memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
            return vector_index.as_chat_engine(
                chat_mode=ChatMode.CONTEXT,
                llm=llm,
                system_prompt=SYS_PROMPT_SIMPLE,
                memory=memory,
            )
        
        case 'RAG':
            query_llm, _ = get_model(model, embed=None, system_prompt=SYS_PROMPT_REACT_QUERY)
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
                        description="Contact list query engine",
                    ),
                ),
            ]
            memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
            return ReActAgent.from_tools(
                tools=tools,
                llm=llm,
                memory=memory,
                max_iterations=30,
                context=SYS_PROMPT_REACT,
                verbose=True,
            )


def query_llm(query: str, history: list, params: dict) -> str:
    print(f'Query: {query}')
    result = make_chat(**params).chat(query, history)
    print(f'Response: {result}')
    return result.response

def params_from_state() -> dict:
    return {param: st.session_state[param] for param in MODEL_PARAMS}

# Data loading (cached)
contacts = load_labels()

# Initialize state variables
for param, value in (MODEL_PARAMS | {'chat_history': []}).items():
    if param not in st.session_state:
        st.session_state[param] = value

# UI elements
st.title(f'Smart contact list')
tabs = st.tabs(["Chatbot", "Contact list", "Parameters"])
with tabs[0]:
    st.header('Chatbot')
    messages = st.container(height=300)
    if query := st.chat_input("Say something"):
        response = query_llm(
            query=query,
            history=st.session_state["chat_history"],
            params=params_from_state())
        if st.session_state["mode"] == 'RAG':
            st.session_state["chat_history"].extend([
                ChatMessage(role=MessageRole.USER, content=query),
                ChatMessage(role=MessageRole.ASSISTANT, content=response),
            ])

    for msg in st.session_state["chat_history"]:
        role = msg.role
        content = msg.content
        with messages.chat_message(name=role):
            st.write(content)

with tabs[1]:
    st.header('Contact list')
    tree_select(contacts)

with tabs[2]:
    st.header('Parameters')
    with st.container(height=480):
        st.selectbox('Mode', ['LLM', 'RAG'], key='mode')
        st.selectbox('Model', MODELS, key='model')
        st.selectbox('Embedder', EMBEDDERS, key='embed')
        st.slider('Temperature', 0.0, 1.0, step=0.1, key='temperature')
        st.slider('Top_p', 0.0, 1.0, step=0.1, key='top_p')
