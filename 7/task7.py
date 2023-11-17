import pandas as pd

data1 = {'ID': [1, 2, 3, 4],
         'Car': ['Ford F-150', 'Chevrolet Corvette', 'Porsche 911', 'Volvo V90']}
df1 = pd.DataFrame(data1)

data2 = {'ID': [1, 3, 2, 5],
         'Price': [43000, 80000, 95000, 55000]}
df2 = pd.DataFrame(data2)

merged_df = pd.merge(df1, df2, on='ID')

print("Объединенный набор данных:")
print(merged_df)
