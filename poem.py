import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.messages.ai import AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

prompt = "Write a small poem about cats"

template = """
    Question: {question}
    Answer: Let's think step by step
"""

llm = Ollama(model="llama3.1")
# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-Nemo-Instruct-2407",
#     temperature=0.5,
#     huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
# )

# https://build.nvidia.com/meta/llama-3_1-405b-instruct?api_key=true
# llm = ChatNVIDIA(
#     model='meta/llama-3.1-405b-instruct',
#     api_key=os.environ['NVIDIA_API_KEY'],
#     base_url='https://integrate.api.nvidia.com/v1',
#     temperature=0.2,
#     top_p=0.7,
#     max_tokens=1024,
# )

res = llm.invoke(prompt)
match res:
    case str() if '\n' in res:
        res = '\n'.join(res.split('\n'))
    case AIMessage():
        res = '\n'.join(res.content.split('\n'))

print(res)