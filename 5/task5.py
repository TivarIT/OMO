import numpy as np

matrix1 = np.random.randint(0, 10, size=(3, 4))
matrix2 = np.random.randint(0, 10, size=(3, 4))

result_matrix = matrix1 + matrix2

print("Первая матрица:")
print(matrix1)
print("\nВторая матрица:")
print(matrix2)
print("\nРезультат сложения:")
print(result_matrix)
