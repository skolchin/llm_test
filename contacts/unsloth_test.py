import os
import re
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

load_dotenv()

P = 'Answer the following question using only provided context. Return only the answer without context or explanations.'
Q = "Кто отвечает за питание сотрудников в Москве?"

PJ = """
    Assert correctness of answer to given question on provided context.
    Return only "correct" or "incorrect" without any explanations.

    Context is {context}.

    Question is {question}.

    Answer is {answer}.
"""

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=".models/llama3.1-trained",
    # model_name='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)
model = FastLanguageModel.for_inference(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

with open('./data/contacts.xml', 'rt') as fp:
    context = fp.read()

inputs = tokenizer.apply_chat_template([
        {'role': 'system', 'content': P}, 
        {'role': 'system', 'content': context},
        {'role': 'user', 'content': Q},
    ], tokenize=False, add_generation_prompt=False)

inputs = tokenizer(inputs, return_tensors='pt').to('cuda')

outputs = model.generate(
    input_ids=inputs.input_ids, 
    attention_mask=inputs.attention_mask,
    max_new_tokens=128, 
    use_cache=True)

prompt_length = inputs.input_ids.shape[1]
answer = tokenizer.decode(outputs[0][prompt_length:])
answer = re.sub(r'<\|.*\|>?', '', answer).replace('\n','')
print(f'\nQ: {Q}\nA: {answer}\n===')

from langchain_community.llms.yandex import YandexGPT
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from lib.yandex_api import get_yandex_iam_token

judge_llm = YandexGPT(
    model_name='yandexgpt',
    folder_id=os.environ['YANDEX_FOLDER_ID'], 
    iam_token=get_yandex_iam_token())

judge_llm = PromptTemplate.from_template(PJ) | judge_llm | StrOutputParser()

A = judge_llm.invoke(context=context, question=Q, answer=answer)
print(f'Assertion: {A}')
