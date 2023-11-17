import pandas as pd

file_path = 'file.csv'
data = pd.read_csv(file_path)
print(data.head())
