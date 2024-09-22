# Prepare Titanic SQLite database from titanic dataset
# Dataset source: https://www.kaggle.com/datasets/vinicius150987/titanic3

import sqlite3
import pandas as pd

df = pd.read_csv("./data/titanic.csv", sep=';')
print(df)

# df.to_sql(index=False, name="titanic", con=sqlite3.connect("./data/titanic.db"), if_exists="replace")

print(f'Number of survived passengers: {len(df.query("survived>0"))}')
print(f'Total number of records: {len(df)}')
