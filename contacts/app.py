# sudo systemctl start ollama
# streamlit run app.py

import os
import json
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from streamlit_tree_select import tree_select
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import XMLReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

load_dotenv()

@st.cache_data
def load_data():
    with open('./../data/labels.json', 'rt') as fp:
        nodes = json.load(fp)
    return nodes

SYS_PROMPT = """
    You are a chatbot, able to have normal interactions, 
    and answer on questions related to contact list.

    Consider that person's name is embedded in <full_name> tag,
    position in <position> tag, department in <department> tag and
    office location in <location> tag.

    You must analyse the provided documents and establish relationship
    among those tags in order to answer questions. For example, 
    for the question 'Who is director of Accounting in Moscow?' you
    must find a person with position equal to 'Director', department
    equal to 'Accounting and office location equal to 'Moscow'.

    If an answer is about a person, always return full person name, job title, 
    deparment and phone number. Multiple results must be provided by one person per line.

    Your answers must be precise, complete and correct.
    If there's not enought information to answer the questions,
    simply state you don't know.
"""
QUERY_WRAPPER = PromptTemplate("{query_str}")

SYS_PROMPT_STABLELM = """<|SYSTEM|># """ + SYS_PROMPT

QUERY_WRAPPER_STABLELM = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

MODELS = {
    'llama3.1': {
        'engine': 'ollama', 
        'model_name': 'llama3.1', 
    }, 
    'mistral-nemo': {
        'engine': 'ollama', 
        'model_name': 'mistral-nemo', 
    },
    'command-r': {
        'engine': 'ollama', 
        'model_name': 'command-r', 
    },
    'saiga-llama3': {
        'engine': 'ollama', 
        'model_name': 'bambucha/saiga-llama3', 
    }, 
    'stablelm': {
        'engine': 'huggingface', 
        'model_name': 'stabilityai/stablelm-2-1_6b-chat', 
        'tokenizer': 'stabilityai/stablelm-2-1_6b-chat',
        'system_prompt': SYS_PROMPT_STABLELM,
        'query_wrapper_prompt': QUERY_WRAPPER_STABLELM,
    },
    'mistral-nemo-cloud': {
        'engine': 'huggingface-cloud', 
        'model_name': 'mistralai/Mistral-Nemo-Instruct-2407', 
        'tokenizer': 'mistralai/Mistral-Nemo-Instruct-2407',
    },
}
EMBEDDERS = {
    'llama3.1': {
        'engine': 'ollama', 
        'model_name': 'llama3.1', 
    },
    'mistral-nemo': {
        'engine': 'ollama', 
        'model_name': 'mistral-nemo', 
    },
    'enbedrus': {
        'engine': 'ollama', 
        'model_name': 'evilfreelancer/enbeddrus', 
    },
    'baai-bge-m3': {
        'engine': 'huggingface', 
        'model_name': 'BAAI/bge-m3', 
    },
}

MODEL_PARAMS = {
    'model':'llama3.1',
    'embed':'llama3.1',
    'temperature': 0.3,
    'top_p': 0.9,
}

@st.cache_resource
def make_chat(model: str, embed: str, temperature: float, top_p: float):
    print(f'Loading model {model} ({temperature}, {top_p})')

    match MODELS[model]['engine']:
        case 'ollama':
            llm = Ollama(
                model=MODELS[model]['model_name'], 
                temperature=temperature, 
                top_p=top_p,
            )

        case 'huggingface':
            import torch
            llm = HuggingFaceLLM(
                model_name=MODELS[model]['model_name'],
                tokenizer_name=MODELS[model].get('tokenizer', MODELS[model]['model_name']),
                context_window=4096,
                max_new_tokens=256,
                generate_kwargs={
                    "temperature": temperature or 0.01, 
                    "do_sample": (temperature > 0.0),
                    "top_p": top_p,
                },
                system_prompt=MODELS[model].get('system_prompt'),
                query_wrapper_prompt=MODELS[model].get('query_wrapper_prompt'),
                device_map="cuda",
                # stopping_ids=[50278, 50279, 50277, 1, 0],
                tokenizer_kwargs={"max_length": 4096},
                model_kwargs={"torch_dtype": torch.float16}
            )

        case 'huggingface-cloud':
            import torch
            llm = HuggingFaceInferenceAPI(
                model_name=MODELS[model]['model_name'],
                token=os.environ['HUGGINGFACE_API_KEY'],
                temperature=temperature,
                top_p=top_p,
            )

    print(f'Loading embedder {embed}')
    match EMBEDDERS[embed]['engine']:
        case 'ollama':
            emded_llm = OllamaEmbedding(model_name=EMBEDDERS[embed]['model_name'])

        case 'huggingface':
            emded_llm = HuggingFaceEmbedding(model_name=EMBEDDERS[embed]['model_name'], device='cuda')

    Settings.llm = llm
    Settings.embed_model = emded_llm

    documents = SimpleDirectoryReader('./../data/', 
                                      required_exts=['.xml'],
                                      file_extractor={'.xml': XMLReader()}).load_data()
    vector_index = VectorStoreIndex.from_documents(documents)

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    chat = vector_index.as_chat_engine(
        chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
        llm=llm,
        system_prompt=SYS_PROMPT,
        memory=memory,
    )
    return chat

def query_llm(query: str, history: list, params: dict) -> str:
    print(f'Query: {query}')
    result = make_chat(**params).chat(query, history)
    print(f'Response: {result}')
    return result.response

def params_from_state() -> dict:
    return {param: st.session_state[param] for param in MODEL_PARAMS}

# Data loading (cached)
contacts = load_data()

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
    with st.container(height=350):
        st.selectbox('Model', MODELS, key='model')
        st.selectbox('Embedder', EMBEDDERS, key='embed')
        st.slider('Temperature', 0.0, 1.0, step=0.1, key='temperature')
        st.slider('Top_p', 0.0, 1.0, step=0.1, key='top_p')
