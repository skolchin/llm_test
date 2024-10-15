# sudo systemctl start ollama
import random
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import XMLReader

from lib.get_model import get_model
from lib.get_prompt import get_prompt

llm, emded_llm = get_model('llama3.1', 'llama3.1', temperature=0.1)
Settings.llm = llm
Settings.embed_model = emded_llm

documents = XMLReader(1).load_data('./data/contacts.xml')
doc_str = "\n".join([x.text for x in random.choices(documents, k=3)])
print(f'Random nodes (3):\n{doc_str}')

Q = 'Кто отвечает за питание сотрудников в Тамани?'

vector_index = VectorStoreIndex.from_documents(documents)
chat = vector_index.as_chat_engine(
    chat_mode='context',
    llm=llm,
    system_prompt=get_prompt('system'),
)
A = chat.chat(Q)
print(f'\nQ: {Q}\nA: {A}')
