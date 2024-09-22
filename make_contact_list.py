import pandas as pd
from faker import Faker
from collections import OrderedDict

JOBS = {
    1: {'name': 'Специалист', 'weight': 0.7},
    2: {'name': 'Старший специалист', 'weight': 0.5}, 
    3: {'name': 'Начальник отдела', 'weight': 0.3}, 
    4: {'name': 'Директор', 'weight': 0.1}, 
}

DEPTS = {
    1: {
        'name': 'Столовая',
        'weight': 0.2,
        'responsible_for': [
            'Питание сотрудников',
            'Завтраки',
            'Обеды',
        ],
    },
    2: {
        'name': 'Бухгалтерия',
        'weight': 0.2,
        'responsible_for': [
            'Заработная плата',
            'Отпускные',
            'Авансовый отчет',
        ],
    },
    3: {
        'name': 'IT support',
        'weight': 0.5,
        'responsible_for': [
            'Выдать новый компьютер',
            'Не работает компьютер',
        ],
    },
}

JOBS_WEIGHTS = OrderedDict([(n, v['weight']) for n, v in JOBS.items()])
DEPTS_WEIGHTS = OrderedDict([(n, v['weight']) for n, v in DEPTS.items()])

faker = Faker('ru_RU')

depts = []
for id in DEPTS:
    dept = DEPTS[id]['name']
    resp = ','.join(DEPTS[id]['responsible_for'])
    depts.append({'name': dept, 'responsible_for': resp})
df_depts = pd.DataFrame(depts)

contacts = []
for _ in range(30):
    name = faker.unique.name()
    phone = faker.unique.phone_number()
    job_id = faker.random_elements(JOBS_WEIGHTS, length=1, use_weighting=True)[0]
    job = JOBS[job_id]['name']
    dept_id = faker.random_elements(DEPTS_WEIGHTS, length=1, use_weighting=True)[0]
    dept = DEPTS[dept_id]['name']
    print(f'{name}, {phone}, {job}, {dept}')
    contacts.append({'full_name': name, 'phone': phone, 'job_title': job, 'department': dept})

df_contacts = pd.DataFrame(contacts)

df_depts.to_xml("./data/departments.xml", root_name='departments', row_name='department', index=False)
df_contacts.to_xml("./data/contacts.xml", root_name='contact_list', row_name='person', index=False)
