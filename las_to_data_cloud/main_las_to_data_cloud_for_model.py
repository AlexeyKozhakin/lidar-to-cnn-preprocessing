import laspy
import numpy as np
import matplotlib.pyplot as plt


# Функция для разбиения точек по сетке и подсчета количества точек в каждом квадрате
def split_into_grid(las_data, grid_size=1):
    x = las_data.x
    y = las_data.y

    # Определяем минимальные и максимальные координаты
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # Создаем сетку
    x_bins = np.arange(min_x, max_x + grid_size, grid_size)
    y_bins = np.arange(min_y, max_y + grid_size, grid_size)

    # Подсчет количества точек в каждом прямоугольнике
    counts, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])

    return counts, x_bins, y_bins


# Функция для визуализации данных
def plot_histogram(counts):
    plt.figure(figsize=(10, 6))
    plt.hist(counts.flatten(), bins=50, edgecolor='black')
    plt.title('Распределение количества точек в квадратах сетки')
    plt.xlabel('Количество точек')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.show()


# Основная функция обработки LAS файла
def process_las_file(file_path, grid_size=1):
    # Открываем LAS файл
    las = laspy.read(file_path)

    # Разбиваем на сетку и получаем количество точек в каждой ячейке
    counts, x_bins, y_bins = split_into_grid(las, grid_size)

    # Показываем информацию о количестве точек в сетке
    print(f'Общее количество прямоугольников: {counts.size}')
    print(f'Максимальное количество точек в одном прямоугольнике: {np.max(counts)}')
    print(f'Минимальное количество точек в одном прямоугольнике: {np.min(counts)}')

    # Строим гистограмму распределения точек
    plot_histogram(counts)


# Пример использования
file_path = (r'C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing'
             r'\data\las_org\data_las_stpls3d\LosAngeles'
             r'\100_-100.las')  # Замените на путь к вашему LAS файлу
# file_path = (r'C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing'
#              r'\data\las_org\data_las_musac_2018_city'
#              r'\437900_3988850.las')  # Замените на путь к вашему LAS файлу
grid_size = 1  # Размер квадрата (можно изменить)

process_las_file(file_path, grid_size)
