# sudo systemctl start ollama
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import TextLoader
from tools import tool_json_to_ppt

model = ChatOllama(model="mistral-nemo")

loader = TextLoader('data/sample_content.txt')
doc = loader.load()

system_message = """ You are an experienced powerpoint designed """

json_mesage = """
    Summarize the main themes from the input below and arrange results in a list of JSON objects.

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
    }}
    Double check that JSON objects are correct and valid.
    Do not output explanations, return only JSON content.

    {doc}
"""
ppt_message = """
    Convert list of JSON objects to powerpoint presentation 
    using 'tool_json_to_ppt' tool.
    
    Output must be a powerpoint presentation file name returned by the tool.
"""

tools = [tool_json_to_ppt]
app = create_react_agent(model, tools)
output = app.invoke({"messages": [
    ("system", system_message),
    ("human", json_mesage.format(doc=doc)),
]})

message_history = output["messages"]
output = app.invoke({"messages": message_history + [('human', ppt_message)]})

print(output['messages'][-1])
