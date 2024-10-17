# sudo systemctl start ollama
# patch required: https://github.com/langchain-ai/langchain/issues/22532
import os
import yaml
from pathlib import Path
from functools import cache
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.llms import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import PromptTemplate
from typing import Tuple, List

from lib.get_prompt import get_prompt

load_dotenv()

@cache
def _load_defs():
    with open(Path(__file__).parent.parent.joinpath('config', 'models.yaml'), 'rt') as fp:
        return yaml.safe_load(fp)

def _get_models():
    return _load_defs()['models']

def _get_embedders():
    return _load_defs()['embedders']

def list_models() -> List[str]:
    return list(_get_models().keys())

def list_embedders() -> List[str]:
    return list(_get_embedders().keys())

def get_system_prompt(model: str) -> str:
    """ Returns a system prompt for given model """
    prompt_key = _get_models()[model].get('system_prompt', 'system')
    return get_prompt(prompt_key)

def get_query_prompt_template(model: str) -> PromptTemplate:
    """ Returns a query wrapper prompt template for given model """
    prompt_key = _get_models()[model].get('query_wrapper_prompt', 'query_wrapper')
    return PromptTemplate(get_prompt(prompt_key))

def get_model(model: str, embed: str | None = None, **kwargs) -> Tuple[LLM, BaseEmbedding | None]:
    print(f'Loading model {model} {kwargs}')

    model_def = _get_models()[model]
    embed_def = _get_embedders()[embed] if embed else None

    match model_def['engine']:
        case 'ollama':
            from llama_index.llms.ollama import Ollama

            llm = Ollama(
                model=model_def['model_name'],
                timeout=1000.0,
                **kwargs,
            )

        case 'huggingface':
            import torch
            from llama_index.llms.huggingface import HuggingFaceLLM

            llm = HuggingFaceLLM(
                model_name=model_def['model_name'],
                tokenizer_name=model_def.get('tokenizer', model_def['model_name']),
                context_window=4096,
                max_new_tokens=256,
                generate_kwargs=kwargs,
                system_prompt=get_system_prompt(model),
                query_wrapper_prompt=get_query_prompt_template(model),
                device_map="cuda",
                tokenizer_kwargs={
                    "max_length": 4096
                },
                model_kwargs={
                    "torch_dtype": torch.float16
                }
            )

        case 'huggingface-cloud':
            from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

            llm = HuggingFaceInferenceAPI(
                model_name=model_def['model_name'],
                token=os.environ['HUGGINGFACE_API_KEY'],
                **kwargs,
            )

        case 'yandex':
            from yandex_cloud_ml_sdk import YCloudML
            from lib.yandex_api import get_yandex_iam_token
            from llama_index.llms.langchain import LangChainLLM

            sdk = YCloudML(
                folder_id=os.environ['YANDEX_FOLDER_ID'], 
                auth=get_yandex_iam_token())
            model = sdk.models.completions(model_def['model_name'])
            model = model.configure(temperature=kwargs.get('temperature', 0.6))
            llm = LangChainLLM(model.langchain())

        case 'unsloth':
            from langchain_community.llms.llamacpp import LlamaCpp
            from llama_index.llms.langchain import LangChainLLM

            model = LlamaCpp(
                model_path=model_def['model_path'],
                n_gpu_layers=64,
                temperature=kwargs.pop('temperature', 0.6),
                model_kwargs={
                    'model_name': model_def['model_name'],
                    'max_seq_length': kwargs.pop('max_seq_length', model_def.get('max_seq_length', 2048)),
                    'load_in_4bit': kwargs.pop('load_in_4bit', model_def.get('load_in_4bit', True)),
                },
                **kwargs
            )
            llm = LangChainLLM(model)

            from transformers import AutoTokenizer
            from llama_index.core import set_global_tokenizer
            Settings.tokenizer = AutoTokenizer.from_pretrained(Path(model_def['model_path']).parent).encode

        case _:
            raise ValueError(model)

    if not embed_def:
        return llm, None

    print(f'Loading embedding {embed}')
    match embed_def['engine']:
        case 'ollama':
            from llama_index.embeddings.ollama import OllamaEmbedding
            emded_llm = OllamaEmbedding(model_name=embed_def['model_name'])

        case 'huggingface':
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            emded_llm = HuggingFaceEmbedding(model_name=embed_def['model_name'], device='cuda')

        case 'yandex':
            from lib.yandex_api import get_yandex_iam_token, YandexGPTEmbeddingExt
            emded_llm = YandexGPTEmbeddingExt(folder_id=os.environ['YANDEX_FOLDER_ID'], api_key=get_yandex_iam_token())

        case 'unsloth':
            from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
            from llama_index.embeddings.langchain import LangchainEmbedding
            embed_model = LlamaCppEmbeddings(
                model_path=embed_def['model_path'], 
                n_gpu_layers=64,
                device='cuda')
            emded_llm = LangchainEmbedding(embed_model)

        case _:
            raise ValueError(embed)

    return llm, emded_llm
