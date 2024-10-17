import re
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

P = 'Answer the following question using only provided context. Return only the answer without context or explanations.'
Q = "Кто отвечает за питание сотрудников в Москве?"

max_seq_length = 2048

with open('./data/contacts.xml', 'rt') as fp:
    context = fp.read()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/llama3.1-trained",
    # model_name='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.for_inference(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

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
A = tokenizer.decode(outputs[0][prompt_length:])
A = re.sub(r'<\|.*\|>?', '', A).replace('\n','')
print(f'\nQ: {Q}\nA: {A}\n===')
