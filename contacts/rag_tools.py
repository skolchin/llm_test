import xml.etree.ElementTree as ET

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
