import json
from langchain.tools import tool
from ppt import create_presentation

@tool(infer_schema=True, parse_docstring=True)
def tool_json_to_ppt(content: str) -> str:
    """ Tool to convert JSON to powerpoint presentation file.

    Args:
        content: JSON to generate presentation from
    """
    js = json.loads(content)
    match js:
        case dict() if not ('sections' in js or 'slides' in js):
            raise ValueError(f'Error: unexpected structure {list(js.keys())}')
        case list() | dict():
            pass
        case _:
            raise ValueError(f'Error: unknown content type {type(js)!r}')

    file_name = create_presentation(js)
    print(f'{file_name} generated')
    return file_name
