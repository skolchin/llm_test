import sys
from .crew import AnyQueryCrew

# QUESTION = "How many records is in 'dbeis_smcntry' table are not for Russia?"
# QUESTION = "Сколько записей в таблице 'dbeis_smcntry' относятся к России? Ответ должен быть на русском языке"
# QUESTION = "How many records in 'dbps21_aadoc' table belongs to document type 'AAA'?"
QUESTION = "What summary amount from 'dbps21_aadoc' table belongs to document type 'AAA'?"

# sudo systemctl start ollama
def run():
    inputs = {
        "question": QUESTION
    }
    AnyQueryCrew().crew().kickoff(inputs=inputs)

def train():
    inputs = {
        "question": QUESTION
    }
    try:
        AnyQueryCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs, filename='train.pkl')
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
    
if __name__ == '__main__':
    run()