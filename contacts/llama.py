# sudo systemctl start ollama
import random
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import XMLReader
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

from lib.get_llama_model import get_model
from lib.get_prompt import get_prompt

llm, emded_llm = get_model('llama3.1-trained', 'llama3.1-trained', temperature=0.1)
Settings.llm = llm
Settings.embed_model = emded_llm

documents = XMLReader(1).load_data('./data/contacts.xml')
# doc_str = "\n".join([x.text for x in random.choices(documents, k=3)])
# print(f'Random nodes (3):\n{doc_str}')

Q = 'Кто отвечает за питание сотрудников в Москве?'

vector_index = VectorStoreIndex.from_documents(documents)
chat = vector_index.as_chat_engine(
    chat_mode='context',
    llm=llm,
    system_prompt=get_prompt('system'),
)
A = chat.chat(Q)
print(f'\nQ: {Q}\nA: {A}')

# eval_llm, _ = get_model('yandex-gpt', temperature=0)
# e1 = FaithfulnessEvaluator(llm=eval_llm)
# e2 = RelevancyEvaluator(llm=eval_llm)

# print(f'Answer is correct? {e1.evaluate_response(query=Q, response=A).feedback}')
# print(f'Answer is relevant? {e2.evaluate_response(query=Q, response=A).feedback}')
