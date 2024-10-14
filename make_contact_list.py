import json
import click
from faker import Faker
from itertools import repeat

@click.command()
@click.argument('locale', type=click.Choice(['en_US', 'ru_RU']), default='en_US')
def main(locale: str):
    faker = Faker(locale)

    with open('./config/contacts.json', 'rt') as fp:
        data = json.load(fp)
        JOBS = data['jobs'][locale]
        DEPTS = data['departments'][locale]
        LOCATIONS = data['locations'][locale]

    indent = lambda n: ''.join(repeat('  ', n))

    output = [
        f"<?xml version='1.0' encoding='utf-8'?>",
        f"<contact_list locale='{locale}'>",
    ]
    labels = []

    loc_id, dept_id, person_id = 1,1,1
    for loc_key in LOCATIONS:
        loc = LOCATIONS[loc_key]["name"]

        # output.append(f'{indent(1)}<location>')
        # output.append(f'{indent(2)}<location_id>{loc_id}</location_id>')
        # output.append(f'{indent(2)}<location_name>{loc}</location_name>')
        # output.append(f'{indent(2)}<departments>')

        labels.append({
            'label': loc,
            'value': [loc_id, None, None],
            'children': []
        })
        loc_label = labels[-1]

        for dept_key in DEPTS:
            dept = DEPTS[dept_key]["name"]

            # output.append(f'{indent(3)}<department>')
            # output.append(f'{indent(4)}<department_id>{dept_id}</department_id>')
            # output.append(f'{indent(4)}<department_name>{dept}</department_name>')
            # output.append(f'{indent(4)}<responsibilities>')
            # for resp in DEPTS[dept_key]["responsible_for"]:
            #     output.append(f'{indent(5)}<responsibility>{resp}</responsibility>')
            # output.append(f'{indent(4)}</responsibilities>')
            # output.append(f'{indent(4)}<staff>')

            loc_label['children'].append({
                'label': dept,
                'value': [loc_id, dept_id, None],
                'children': []
            })
            dept_label = loc_label['children'][-1]

            for job_key in reversed(JOBS):
                job = JOBS[job_key]["name"]

                for _ in range(faker.random_int(1, int(JOBS[job_key]['max_count']))):
                    full_name = faker.unique.name()
                    phone = faker.unique.phone_number()

                    output.append(f'{indent(1)}<person>')
                    output.append(f'{indent(2)}<person_id>{person_id}</person_id>')
                    output.append(f'{indent(2)}<full_name>{full_name}</full_name>')
                    output.append(f'{indent(2)}<phone>{phone}</phone>')
                    output.append(f'{indent(2)}<position>{job}</position>')
                    output.append(f'{indent(2)}<department>{dept}</department>')
                    output.append(f'{indent(2)}<location>{loc}</location>')
                    output.append(f'{indent(2)}<responsibilities>')
                    for resp in DEPTS[dept_key]["responsible_for"]:
                        output.append(f'{indent(3)}<responsibility>{resp}</responsibility>')
                    output.append(f'{indent(2)}</responsibilities>')
                    output.append(f'{indent(1)}</person>')

                    dept_label['children'].append({
                        'label': f'{full_name}, {job}, {phone}',
                        'value': [loc_id, dept_id, person_id],
                    })
                    person_id += 1

            # output.append(f'{indent(4)}</staff>')
            # output.append(f'{indent(3)}</department>')
            dept_id += 1
            
        # output.append(f'{indent(2)}</departments>')
        # output.append(f'{indent(1)}</location>')
        loc_id += 1

    output.append('</contact_list>')

    with open('./data/contacts.xml', 'wt') as fp:
        fp.writelines([line + '\n' for line in output])

    with open('./data/labels.json', 'wt') as fp:
        json.dump(labels, fp, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()