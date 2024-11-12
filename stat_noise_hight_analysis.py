import laspy
import numpy as np
import matplotlib.pyplot as plt

# Шаг 1: Чтение LAS-файла
# las_file = laspy.read(
#     r"C:\Users\alexe\Downloads\UM\work\Data"
#     r"\2012-20240803T112213Z-001\2018-20240913T181437Z-001"
#     r"\city\city\437650_3988700.las")
'''
las_file = laspy.read(
    r"D:\data\las_org\data_las_stpls3d\all_org_las_cut_256"
    r"\4_points_GTv2_0_0.las")
'''
las_file = laspy.read(
    r"D:\data\las_org\san_gwan_256_256_1"
    r"\453632_3974144.las")   
 
    # Выводим количество точек в файле
print(f"Количество точек в LAS файле: {len(las_file.points)}")
'''
# Шаг 2: Извлечение координат высоты (Z)
z_coordinates = las_file.z

# Шаг 3: Построение гистограммы высот
plt.figure(figsize=(10, 6))
plt.hist(z_coordinates, bins=50, color='skyblue', edgecolor='black')

# Шаг 4: Настройка графика
plt.title("Гистограмма высот точек", fontsize=16)
plt.xlabel("Высота (метры)", fontsize=14)
plt.ylabel("Количество точек", fontsize=14)

# Шаг 5: Показать гистограмму
plt.grid(True)
plt.show()
'''
'''
# Шаг 2: Извлечение координат высоты (Z)
z_coordinates = las_file.z

# Шаг 3: Вычисление среднего и стандартного отклонения
mean_z = np.mean(z_coordinates)
std_z = np.std(z_coordinates)

# Шаг 4: Фильтрация данных на основе 3 сигм
z_filtered = z_coordinates[(z_coordinates >= mean_z - 3 * std_z) & (z_coordinates <= mean_z + 3 * std_z)]

# Шаг 5: Построение гистограммы с отфильтрованными данными
plt.figure(figsize=(10, 6))
plt.hist(z_filtered, bins=50, color='skyblue', edgecolor='black')

# Шаг 6: Настройка графика
plt.title("Гистограмма высот точек (с 3-сигмовой фильтрацией)", fontsize=16)
plt.xlabel("Высота (метры)", fontsize=14)
plt.ylabel("Количество точек", fontsize=14)

# Шаг 7: Показать гистограмму
plt.grid(True)
plt.show()
'''