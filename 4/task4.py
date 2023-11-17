import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['class'] = iris.target_names[iris.target]

grouped_data = data.groupby('class')['sepal length (cm)'].mean()

print("Среднее значение длины чашелистика для каждого класса:", grouped_data)