# Test LLM with PowerPoint slides generation

import os
import json
import crewai
from crewai_tools import tool
from dotenv import load_dotenv
# from langchain_ollama.llms import OllamaLLM
from langchain_community.llms import Ollama
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEndpoint

from ppt import create_presentation

load_dotenv()
# set_llm_cache(SQLiteCache(database_path=".langchain.db"))

llm = Ollama(model="mistral-nemo")

# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-Nemo-Instruct-2407",
#     huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
#     temperature=0.35,
# )

# https://build.nvidia.com/meta/llama-3_1-405b-instruct
# llm = ChatNVIDIA(
#     model='meta/llama-3.1-405b-instruct',
#     api_key=os.environ['NVIDIA_API_KEY'],
#     base_url='https://integrate.api.nvidia.com/v1',
#     temperature=0.2,
#     top_p=0.7,
#     max_tokens=1024,
# )

@tool('tool_validate_json')
def tool_validate_json(content: str | dict | list) -> str:
    """ Validates JSON content.

    Args:
        content: JSON content to validate
    """
    match content:
        case str():
            try:
                json.loads(content)
            except Exception as ex:
                return f'Error: {ex}'
            
        case dict():
            if not ('sections' in content or 'slides' in content):
                return f'Error: unexpected structure {list(content.keys())}'
        
        case list():
            if len(content) < 2:
                return f'Error: list must be at least 2 elements long, got {len(content)}'
        
        case _:
            return f'Error: only certain types are supported, got {type(content)!r} instead'
        
    return 'ok'

@tool('tool_json_to_ppt')
def tool_json_to_ppt(content: str | dict | list) -> str:
    """ Convert array of JSON to powerpoint presentation and save it to file.

    Args:
        content: JSON content to generate presentation from
    """
    match content:
        case str():
            content = json.loads(content)
        case dict():
            if 'sections' in content:
                content = content['sections']
            elif 'slides' in content:
                content = content['slides']
        case list():
            pass

    try:
        return create_presentation(content)
    except Exception as ex:
        return f'Error: {ex}'

tool_json_to_ppt.result_as_answer = True

with open('./data/sample_content.txt', 'rt') as fp:
    content = fp.read()

SYS_PROMPT = '''Create powerpoint presentation file based on {user_input}.'''

# prompt = f'[INST] {SYS_PROMPT2} [/INST]\n{content}'
# res = llm.invoke(prompt)
# print(res)

agent_writer = crewai.Agent(
    role="Powerpoint designer",
    goal=SYS_PROMPT,
    backstory='''
        You are an experienced powepoint designer.

        Summarize the input text, arrange it in a list of JSON objects,
        then convert JSON objects list to powerpoint presentation file.

        Determine the needed number of json objects (slides) based on the length of the text. 
        Each key point in a slide should be limited to up to 10 words. 
        Consider maximum of 5 bullet points per slide. 
        The first item in the list must be a json object for the title slide. 
        This is a sample of title slide json object:
        {{
            "id": 1,
            "title_text": "My Presentation Title",
            "subtitle_text": "My presentation subtitle",
            "is_title_slide": "yes"
        }}
        Here is the sample of json data for other slides:
        {{
            "id": 2, 
            title_text": "Slide 1 Title", 
            "text": ["Bullet 1", "Bullet 2"]
        }},
        {{
            "id": 3, 
            title_text": "Slide 2 Title", 
            "text": ["Bullet 1", "Bullet 2", "Bullet 3"],
            "img_path": ["Illustration 1.png", "Illustration 2.png"]
        }}
        Double check that JSON objects are correct and valid.

        Use tool 'tool_validate_json' to review JSON objects list. Provide the tool with JSON to validate 
        in positional argument 'content'. If JSON objects are invalid, the tool will return an error.
        If such an error is returned, recreate JSON objects, check them, and try again. 

        Use tool 'tool_json_to_ppt' to convert list of JSON objects to powerpoint presentation file.
        If JSON objects list is invalid, the tool will return an error message. 
        If such an error is returned, recreate JSON objects, validate them, and try again.

        Don't output explanation. Do not output JSON objects. Return only powerpoint presentation file name.
    ''',
    tools=[tool_validate_json, tool_json_to_ppt],
    max_iter=100,
    llm=llm,
    allow_delegation=False, verbose=True)

task_writer = crewai.Task(
    description=SYS_PROMPT,
    agent=agent_writer,
    expected_output='''Powerpoint presentation file name''')

crew = crewai.Crew(
    agents=[agent_writer], 
    tasks=[task_writer],
    memory=True,
    verbose=True,
    embedder={
        "provider": "ollama",
        "config":{
            "model": 'mistral-nemo',
        }
    })

res = crew.kickoff(inputs={"user_input": content})
print(f'Results: {res}')
