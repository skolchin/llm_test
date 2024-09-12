import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# llm = Ollama(model="llama3.1")
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
    huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
    temperature=0.2,
    top_p=0.7,
)

SYS_PROMPT = """
    Summarize the input text and arrange it in an array of JSON objects to to be suitable for a powerpoint presentation. 
    Determine the needed number of json objects (slides) based on the length of the text. 
    Each key point in a slide should be limited to up to 10 words. 
    Consider maximum of 5 bullet points per slide. 
    Return the response as an array of json objects. 
    The first item in the list must be a json object for the title slide. 
    This is a sample of such json object:
        {
        "id": 1,
        "title_text": "My Presentation Title",
        "subtitle_text": "My presentation subtitle",
        "is_title_slide": "yes"
        }
    And here is the sample of json data for slides:
        {"id": 2, title_text": "Slide 1 Title", "text": ["Bullet 1", "Bullet 2"]},
        {"id": 3, title_text": "Slide 2 Title", "text": ["Bullet 1", "Bullet 2", "Bullet 3"]}

    Make sure the json object is correct and valid. 
    Don't output explanation. I just need the JSON array as your output.
"""

with open('content.txt', 'rt') as fp:
    content = fp.read()

prompt = SYS_PROMPT + f"\n{content}"

res = llm.invoke(prompt)
print(res)