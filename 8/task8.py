import numpy as np

matrix = np.random.randint(0, 10, size=(4, 4))
vector = np.random.randint(0, 10, size=4)

result_vector = np.dot(matrix, vector)

print("Матрица:")
print(matrix)
print("\nВектор:")
print(vector)
print("\nРезультат умножения матрицы на вектор:")
print(result_vector)
