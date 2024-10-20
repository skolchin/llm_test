# sudo systemctl start ollama
# https://blog.gopenai.com/fine-tuning-llama-3-with-unsloth-unleashing-speed-and-efficiency-3cde835ef531
import torch
import random
from trl import SFTTrainer
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from unsloth.chat_templates import get_chat_template

max_seq_length = 512

with open('./data/contacts.xml', 'rt') as fp:
    context = fp.read()

prompt_with_context = f'Answer to user question relying only on this context:\n{context}'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

def formatting_prompts_func(examples):
    sys_content = [ {'role': 'system', 'content': prompt_with_context} ] * len(examples['0'])
    texts = [tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=False) 
             for x in zip(sys_content, examples['0'], examples['1'])]
    return { "text" : texts }

dataset = Dataset.from_json('./data/training.json')
dataset = dataset.map(formatting_prompts_func, batched = True)

# doc_str = "\n".join([x['text'] + '\n' for x in random.choices(dataset, k=3)])
# print(f'Random nodes (3):\n{doc_str}')

ds_splits = dataset.train_test_split(test_size=0.2)
print(f'Train/val length: {ds_splits["train"].num_rows}/{ds_splits["test"].num_rows}')

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    use_rslora=False,
    loftq_config=None,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds_splits["train"],
    eval_dataset=ds_splits["test"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing=True,
    eval_packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=50,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False, #not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir="./.train",
        eval_strategy='steps',
        save_strategy='steps',
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
    ),
    # callbacks=[EarlyStoppingCallback(5)]
)

trainer.train()

model = FastLanguageModel.for_inference(model)
model.save_pretrained('./models/llama3.1-trained')
model.save_pretrained_gguf("./models/llama3.1-trained", tokenizer, quantization_method="quantized")
# model.save_pretrained_merged('./models/llama3.1-trained', tokenizer, save_method="merged_16bit")
