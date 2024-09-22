from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding

llm = Ollama(model="mistral-nemo")
emded_llm = OllamaEmbedding(model_name="llama3.1")

Settings.llm = llm
Settings.embed_model = emded_llm

documents = SimpleDirectoryReader('./data/', required_exts=['.xml']).load_data()

vector_index = VectorStoreIndex.from_documents(documents)
# query_engine = vector_index.as_query_engine(llm=llm)

SYS_PROMPT = ("""
    You are a chatbot, able to have normal interactions, 
    and answer on questions related to contact list.
                
    If a question is related to personnel responsibilities,
    always return full person name, deparment and phone number.
    Multiple results must be provided in one person per line.
              
    Your answers must be precise, complete and correct.
    If there's not enought information to answer the questions,
    simply state you don't know.
    """
)

chat = vector_index.as_chat_engine(
    chat_mode='context',
    llm=llm,
    system_prompt=SYS_PROMPT,
)
result = chat.chat('Кто в ответе за бюджет?')
print(result)
