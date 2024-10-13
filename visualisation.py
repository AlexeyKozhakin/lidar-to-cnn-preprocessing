import laspy
import numpy as np

# Загружаем LAS-файл
las_file_path = 'data_las_musac_2018_colored/musac_2018_size500x500.las'
las = laspy.read(las_file_path)

# Извлекаем координаты точек
points = np.vstack((las.x, las.y, las.z)).transpose()

# Выведем базовую информацию
print(f"Количество точек: {len(points)}")

import open3d as o3d

# Создание объекта PointCloud для визуализации
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Визуализация 3D-точечного облака
o3d.visualization.draw_geometries([pcd])
