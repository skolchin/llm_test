# Prepare Titanic SQLite database from titanic dataset
# Datast source: https://www.kaggle.com/datasets/vinicius150987/titanic3

import pandas as pd
import sqlite3

df = pd.read_csv("titanic.csv", sep=';')
print(df)

# df.to_sql(index=False, name="titanic", con=sqlite3.connect("titanic.db"), if_exists="replace")

print(f'Number of survived passengers: {len(df.query("survived>0"))}')
print(f'Total number of records: {len(df)}')
