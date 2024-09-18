import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = "Write a small poem about cats. Output must be in Russian language."

# sudo systemctl start ollama
llm = Ollama(model="mistral-nemo")

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

llm = llm | StrOutputParser()

res = llm.invoke(prompt)
print(res)