import sys
from .crew import AnyQueryCrew

QUESTION = "Сколько записей в таблице 'transl_budget' относятся к первой версии бюджета?"

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