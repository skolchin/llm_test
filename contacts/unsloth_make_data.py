# sudo systemctl start ollama
# https://docs.llamaindex.ai/en/stable/examples/finetuning/mistralai_fine_tuning/
import json
import random
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import XMLReader
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

from lib.get_model import get_model
from lib.get_prompt import get_prompt

llm, emded_llm = get_model('yandex-gpt', 'baai-bge-m3', temperature=0.35)
Settings.llm = llm
Settings.embed_model = emded_llm

documents = XMLReader(1).load_data('./data/contacts.xml')
doc_str = "\n".join([x.text + '\n' for x in random.choices(documents, k=3)])
print(f'Random nodes (3):\n{doc_str}')

vector_index = VectorStoreIndex.from_documents(documents)
query_engine = vector_index.as_query_engine(llm=llm)

dataset_generator = RagDatasetGenerator.from_documents(
    documents,
    llm=llm,
    question_gen_query=get_prompt('fine_tune_generator_yandex'),
    num_questions_per_chunk=3,
)
questions = dataset_generator.generate_questions_from_nodes()
print(f'{len(questions.examples)} questions generated\n')

messages = []
for n, question in enumerate(questions.examples):
    q = question.query.strip('"')
    a = str(query_engine.query(q))
    messages.append([
        {'role': 'user', 'content': q},
        {'role': 'assistant', 'content': a},
    ])
    print(f'Q #{n+1}: {q}\nA: {a}\n===')

with open('./data/training.json', 'wt') as fp:
    json.dump(messages, fp, ensure_ascii=False, indent=2)
