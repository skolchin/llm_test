# Test LLM

import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

QUESTION = '''
    If how many of people had survived and how many people were in total on Titanic?
'''

TEMPLATE = """
    Question: {question}
    Answer: Answer briefly and precisely
"""

# sudo systemctl start ollama
# llm = Ollama(model="llama3.1")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
    huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
    temperature=0.2,
    top_p=0.7,
)

# llm = ChatNVIDIA(
#     model='meta/llama-3.1-405b-instruct',
#     api_key=os.environ['NVIDIA_API_KEY'],
#     base_url='https://integrate.api.nvidia.com/v1',
#     temperature=0.2,
#     top_p=0.7,
#     max_tokens=1024,
# )

prompt = PromptTemplate.from_template(TEMPLATE)
llm_chain = prompt | llm
res = llm_chain.invoke({'question': QUESTION})

match res:
    case str() if '\n' in res:
        res = '\n'.join(res.split('\n'))
    case AIMessage():
        res = '\n'.join(res.content.split('\n'))

print(res)
