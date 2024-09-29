import xml.etree.ElementTree as ET
from textwrap import dedent

SYS_PROMPT_REACT = dedent("""
    You are a chatbot, able to have normal interactions and to answer on questions related to contact list.

    Use available tools to construct an answer for user question.

    Use 'search_contact_list' tool if other tools are inappropriate to answer user question
    or they have returned 'Not found' answer.
                                                    
    If you have used 'search_contact_list' tool and its output contains
    either department or person IDs, stop using this tool and call any other relevant tool.
 
    Final answer must be properly formatted as a table with single person information per line.
    Each person information should contain at least full person name, phone number, position, deparment and office location.
              
    Your answers must be brief, precise, complete and correct. If a user
    questions is not related to contact list search, politely refuse to answer.
                          
    If there's not enought information to answer the questions, report that you don't know.
    """
)

SYS_PROMPT_REACT_QUERY = dedent("""
    You are a query agent specialized on finding information in a contact list.
                                
    You are to find office locations, departments or persons which are most
    relevant to the user query.
                                
    You must return structured answer containing either department IDs or person IDs. 
    Always specify what type of information you have found. 
    If multiple IDs were found, return them as a comma-separated list.

    For example:
        department_id=1, department_id=2
    or:
        person_id=10, person_id=20
    
    If no relevant information was found, return 'Not found'.
    Do not output any explanations.
""")

def find_person_impl(data: ET.Element, name: str) -> str:
    if person_nodes := data.findall(f".//person/full_name[.='{str(name).lower().strip()}']../person_id"):
        return ','.join([f'{a.tag}={a.text}' for a in person_nodes])
    
    return 'Not found'

def get_person_details_impl(data: ET.Element, person_id: str | int | list) -> str:
    if isinstance(person_id, str):
        person_id = [s.strip() for s in person_id.strip('[]').split(',')]
    if not isinstance(person_id, list):
        person_id = [person_id]

    attrs = []
    for i in person_id:
        if isinstance(i, str) and i.startswith('person_id='):
            i = i.split('=')[-1].strip()
        if person_nodes := data.findall(f".//person/person_id[.='{i}'].."):
            dept_nodes = data.findall(f".//person/person_id[.='{i}']../../..")
            loc_nodes = data.findall(f".//person/person_id[.='{i}']../../../../..")
            attrs += [[a for p in person_nodes for a in p] + \
                     [a for p in dept_nodes for a in p if a.tag == 'department_name'] + \
                     [a for p in loc_nodes for a in p if a.tag == 'location_name']]
    
    if not attrs:
        return 'Not found'
    
    return '\n'.join([','.join([f"{b.tag}={b.text}" for b in a]) for a in attrs])

def find_departments_impl(data: ET.Element, dept: str) -> str:
    if ',' in dept:
        dept, office = dept.split(',', 1)
        if dept_nodes := data.findall(f".//location/location_name[.='{office.lower().strip()}']../departments/department/department_name[.='{dept.lower().strip()}']../department_id"):
            a = dept_nodes[0]
            return f'{a.tag}={a.text}'

    if dept_nodes := data.findall(f".//department/department_name[.='{dept.lower()}']../department_id"):
        return ','.join([f'{a.tag}={a.text}' for a in dept_nodes])

    return 'Not found'

def get_department_staff_impl(data: ET.Element, dept_id: str | int | list) -> str:
    if isinstance(dept_id, str):
        dept_id = [s.strip() for s in dept_id.strip('[]').split(',')]
    if not isinstance(dept_id, list):
        dept_id = [dept_id]

    attrs = []
    for i in dept_id:
        if isinstance(i, str) and i.startswith('department_id='):
            i = i.split('=')[-1].strip()
        if staff_nodes := data.findall(f".//department/department_id[.='{i}']../staff"):
            attrs += [b for p in staff_nodes for a in p for b in a if b.tag == 'person_id']

    return ', '.join([f"{a.tag}={a.text}" for a in attrs]) if attrs else 'Not found'
