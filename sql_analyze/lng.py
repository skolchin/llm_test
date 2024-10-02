import os
from pprint import pprint
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# sudo systemctl start ollama
llm = OllamaLLM(model="qwen2.5", temperature=0.1)

# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-Nemo-Instruct-2407",
#     temperature=0.5,
#     huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
# )

# https://build.nvidia.com/meta/llama-3_1-405b-instruct?api_key=true
# llm = ChatNVIDIA(
#     model='meta/llama-3.1-405b-instruct',
#     api_key=os.environ['NVIDIA_API_KEY'],
#     base_url='https://integrate.api.nvidia.com/v1',
#     temperature=0.2,
#     top_p=0.7,
#     max_tokens=1024,
# )

TEMPLATE = """
    statement: {st}
    answer: Analyze SQL statement and provide a list of selected fields
            and a list of tables used by the query.
            For each field indicate which table this field is selected from.
            If a field is calculated by some formula, return the formula as well.
            Answer must be provided as JSON list in the following format:
                {{
                    "fields": [
                        {{
                            "field_name": "xxx",
                            "table_name": "table",
                            "table_alias": "t"
                            "formula": "t.x + 1"
                        }}
                    ],
                    "tables": [
                        {{
                            "table_name": "table",
                            "table_alias": "t"
                        }}
                    ]
                }}
            Do not output explanations.
"""

SQL = """
    select distinct on (job_id)
        j.uuid as job_id,
        r.uuid as run_id,
        rf.facet #>> '{airflow,dag,dag_id}' as dag,
        r.job_name as task,
        r.created_at,
        jm.io_type as usage,
        d.uuid as table_id,
        d.name as table_name,
        trg.uuid as target_table_id,
        trg.name as target_table,
        jf.facet #>> '{sql,query}' as query
    from jobs j
    inner join job_versions_io_mapping jm on jm.job_uuid = j.uuid
    inner join datasets d on jm.dataset_uuid = d.uuid 
    inner join runs r on j.current_version_uuid = r.job_version_uuid 
    left join job_facets jf on jf.job_uuid = j.uuid and jf.name = 'sql' 
    left join run_facets rf on rf.run_uuid = r.uuid and rf.name = 'airflow' and rf.lineage_event_type = 'COMPLETE'
    order by job_id, dag, task, created_at desc;
"""

prompt = PromptTemplate.from_template(TEMPLATE)
llm_chain = prompt | llm | JsonOutputParser()
res = llm_chain.invoke({'st': SQL})
pprint(res, indent=2)
