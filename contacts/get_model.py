# sudo systemctl start ollama
import os
from prompts import get_prompt
from dotenv import load_dotenv
from llama_index.core import PromptTemplate

load_dotenv()

MODELS = {
    'llama3.1': {
        'engine': 'ollama', 
        'model_name': 'llama3.1', 
    }, 
    'mistral-nemo': {
        'engine': 'ollama', 
        'model_name': 'mistral-nemo', 
    },
    'saiga-llama3': {
        'engine': 'ollama', 
        'model_name': 'bambucha/saiga-llama3', 
    }, 
    'stablelm': {
        'engine': 'huggingface', 
        'model_name': 'stabilityai/stablelm-2-1_6b-chat', 
        'tokenizer': 'stabilityai/stablelm-2-1_6b-chat',
        'system_prompt_key': 'system_tagged',
        'query_wrapper_prompt': 'query_wrapper_tagged',
    },
    'mistral-nemo-cloud': {
        'engine': 'huggingface-cloud', 
        'model_name': 'mistralai/Mistral-Nemo-Instruct-2407', 
        'tokenizer': 'mistralai/Mistral-Nemo-Instruct-2407',
    },
    'yandex-gpt': {
        'engine': 'yandex', 
        'model_name': 'yandexgpt', 
        'system_prompt': 'yandex',
    }
}
EMBEDDERS = {
    'llama3.1': {
        'engine': 'ollama', 
        'model_name': 'llama3.1', 
    },
    'mistral-nemo': {
        'engine': 'ollama', 
        'model_name': 'mistral-nemo', 
    },
    'enbedrus': {
        'engine': 'ollama', 
        'model_name': 'evilfreelancer/enbeddrus', 
    },
    'baai-bge-m3': {
        'engine': 'huggingface', 
        'model_name': 'BAAI/bge-m3', 
    },
    'yandex-gpt': {
        'engine': 'yandex', 
        'model_name': 'evilfreelancer/enbeddrus', 
    },
}

def get_system_prompt(model: str) -> str:
    """ Returns a system prompt for given model """
    prompt_key = MODELS[model].get('system_prompt', 'system')
    return get_prompt(prompt_key)

def get_query_prompt_template(model: str) -> PromptTemplate:
    """ Returns a query wrapper prompt template for given model """
    prompt_key = MODELS[model].get('query_wrapper_prompt', 'query_wrapper')
    return PromptTemplate(get_prompt(prompt_key))

def get_model(model: str, embed: str | None = None, **kwargs):
    print(f'Loading model {model} {kwargs}')

    match MODELS[model]['engine']:
        case 'ollama':
            from llama_index.llms.ollama import Ollama

            llm = Ollama(
                model=MODELS[model]['model_name'],
                timeout=1000.0,
                **kwargs,
            )

        case 'huggingface':
            import torch
            from llama_index.llms.huggingface import HuggingFaceLLM

            llm = HuggingFaceLLM(
                model_name=MODELS[model]['model_name'],
                tokenizer_name=MODELS[model].get('tokenizer', MODELS[model]['model_name']),
                context_window=4096,
                max_new_tokens=256,
                generate_kwargs=kwargs,
                system_prompt=get_system_prompt(model),
                query_wrapper_prompt=get_query_prompt_template(model),
                device_map="cuda",
                tokenizer_kwargs={"max_length": 4096},
                model_kwargs={"torch_dtype": torch.float16}
            )

        case 'huggingface-cloud':
            from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

            llm = HuggingFaceInferenceAPI(
                model_name=MODELS[model]['model_name'],
                token=os.environ['HUGGINGFACE_API_KEY'],
                **kwargs,
            )

        case 'yandex':
            from yandex_cloud_ml_sdk import YCloudML
            from yandex_api import get_yandex_iam_token
            from llama_index.llms.langchain import LangChainLLM

            sdk = YCloudML(folder_id=os.environ['YANDEX_FOLDER_ID'], auth=get_yandex_iam_token())
            model = sdk.models.completions(MODELS[model]['model_name'])
            model = model.configure(temperature=kwargs.get('temperature', 0.6))
            llm = LangChainLLM(model.langchain())

        case _:
            raise ValueError(model)

    if not embed:
        return llm, None

    print(f'Loading embedder {embed}')
    match EMBEDDERS[embed]['engine']:
        case 'ollama':
            from llama_index.embeddings.ollama import OllamaEmbedding
            emded_llm = OllamaEmbedding(model_name=EMBEDDERS[embed]['model_name'])

        case 'huggingface':
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            emded_llm = HuggingFaceEmbedding(model_name=EMBEDDERS[embed]['model_name'], device='cuda')

        case 'yandex':
            from yandex_api import get_yandex_iam_token, YandexGPTEmbeddingExt
            emded_llm = YandexGPTEmbeddingExt(folder_id=os.environ['YANDEX_FOLDER_ID'], api_key=get_yandex_iam_token())

        case _:
            raise ValueError(embed)

    return llm, emded_llm

