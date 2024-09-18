import sys
from .crew import TitanicCrew

QUESTION = "How many people are in total in Titanic database and how many of them had died?"

# sudo systemctl start ollama
def run():
    inputs = {
        "question": QUESTION
    }
    TitanicCrew().crew().kickoff(inputs=inputs)

def train():
    inputs = {
        "question": QUESTION
    }
    try:
        TitanicCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs, filename='train.pkl')
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
    
if __name__ == '__main__':
    run()