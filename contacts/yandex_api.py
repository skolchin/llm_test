import os
import aiohttp
import requests
from functools import cache
from dotenv import load_dotenv
from llama_index.embeddings.yandexgpt.util import YException
from llama_index.embeddings.yandexgpt import YandexGPTEmbedding
from tenacity import Retrying, RetryError, stop_after_attempt, wait_fixed
from llama_index.embeddings.yandexgpt.base import DEFAULT_YANDEXGPT_API_BASE
from typing import List

load_dotenv()

@cache
def get_yandex_iam_token() -> str:
    response = requests.post('https://iam.api.cloud.yandex.net/iam/v1/tokens',
                             json={ 'yandexPassportOauthToken': os.environ["YANDEX_OAUTH_TOKEN"] })
    response.raise_for_status()
    token = response.json()['iamToken']
    return token

@cache
def get_yandex_api_key() -> str:
    return os.environ["YANDEX_API_KEY"]

class YandexGPTEmbeddingExt(YandexGPTEmbedding):
    """ Yandex embedder that uses IAM authorization instead of API Key"""

    def _embed(self, text: str, is_document: bool = False) -> List[float]:
        payload = {"modelUri": self._getModelUri(is_document), "text": text}
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-data-logging-enabled": "false",
            "x-folder-id": f"{self.folder_id}",
        }
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(self.retries),
                wait=wait_fixed(self.sleep_interval),
            ):
                with attempt:
                    response = requests.post(
                        DEFAULT_YANDEXGPT_API_BASE, json=payload, headers=header
                    )
                    response = response.json()
                    if "embedding" in response:
                        return response["embedding"]
                    raise YException(f"No embedding found, result returned: {response}")
        except RetryError:
            raise YException(
                f"Error computing embeddings after {self.retries} retries. Result returned:\n{response}"
            )

    async def _aembed(self, text: str, is_document: bool = False) -> List[float]:
        payload = {"modelUri": self._getModelUri(is_document), "text": text}
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-data-logging-enabled": "false",
            "x-folder-id": f"{self.folder_id}",
        }
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(self.retries),
                wait=wait_fixed(self.sleep_interval),
            ):
                with attempt:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            DEFAULT_YANDEXGPT_API_BASE, json=payload, headers=header
                        ) as response:
                            result = await response.json()
                            if "embedding" in result:
                                return result["embedding"]
                            raise YException(
                                f"No embedding found, result returned: {result}"
                            )
        except RetryError:
            raise YException(
                f"Error computing embeddings after {self.retries} retries. Result returned:\n{result}"
            )

