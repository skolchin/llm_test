import yaml
from pathlib import Path
from functools import cache

def _hide_brakets(s: str) -> str:
    for a, b in [('\\{', '--##--'), ('\\}', '--$$--')]:
        s = s.replace(a,b)
    return s

def _unhide_brakets(s: str) -> str:
    for a, b in [('{', '--##--'), ('}', '--$$--')]:
        s = s.replace(b,a)
    return s

@cache
def _load_defs():
    with open(Path(__file__).parent.parent.joinpath('config', 'prompts.yaml'), 'rt') as fp:
        content = yaml.safe_load(fp)
    result = {}
    for k, v in content['prompts'].items():
        x = _unhide_brakets(_hide_brakets(v).format(**content['prompts']))
        result[k] = x
    return result

def get_prompt(key: str) -> str:
    """ Get a prompt for given key """
    return _load_defs()[key]
