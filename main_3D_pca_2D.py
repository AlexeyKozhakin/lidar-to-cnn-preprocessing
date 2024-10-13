import laspy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Чтение LAS файла
las_file = (r"data\las_colored\stpls3d_tenser_128"
            r"\data_las_musac_2018_colored_500x500_san_gwan"
            r"\453000_3974000.las")
las = laspy.read(las_file)

# Извлечение координат x, y, z
x = las.x
y = las.y
z = las.z

# Собираем координаты в один массив
coords = np.vstack((x, y, z)).T

# Нормализация координат (Min-Max нормализация в диапазон [0, 1])
scaler = MinMaxScaler()
coords_normalized = scaler.fit_transform(coords)

# Применение PCA для понижения размерности до 2D
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(coords_normalized)

# Извлечение компонент после PCA
u = coords_2d[:, 0]
v = coords_2d[:, 1]

# Построение графика
plt.figure(figsize=(10, 7))
sc = plt.scatter(u, v, c=z, cmap='viridis', s=1)  # s=1 для размера точек
plt.colorbar(sc, label='Z value')
plt.title('2D PCA Projection with Z as Color')
plt.xlabel('U (PCA Component 1)')
plt.ylabel('V (PCA Component 2)')
plt.show()