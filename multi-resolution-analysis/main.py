import os
import numpy as np
import laspy
from concurrent.futures import ProcessPoolExecutor
import scipy.stats as stats
from collections import Counter

# Функция для смещения точек по оси Z
def shift_z_to_zero(points):
    z_min = np.min(points[:, 2])
    points[:, 2] -= z_min  # Смещаем все точки по оси Z так, чтобы zmin был равен 0
    return points

# Функция для группировки точек и вычисления статистик по квадратам 1x1 метра
def calculate_statistics_and_class(points, classes, grid_size=1):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Определение диапазонов для разбиения по сетке
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # Создание сетки
    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)

    stats_list = []
    class_list = []

    # Для каждого квадрата в сетке
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            # Фильтрация точек, попадающих в данный квадрат
            mask = (x >= x_bins[i]) & (x < x_bins[i + 1]) & (y >= y_bins[j]) & (y < y_bins[j + 1])
            z_values = z[mask]
            class_values = classes[mask]

            if len(z_values) > 0:
                # Рассчет статистических параметров
                z_mean = np.mean(z_values)
                z_std = np.std(z_values)
                z_max = np.max(z_values)
                z_min = np.min(z_values)
                dz = z_max-z_min

                # z_asm = stats.skew(z_values)  # Ассиметрия
                # z_kur = stats.kurtosis(z_values)  # Куртозис

                # Формируем массив (n, n, k), где k - количество статистических параметров
                #stats_list.append([z_mean, z_std, z_asm, z_kur])
                stats_list.append([z_mean, z_std, dz])

                # Находим самый частый класс (мода)
                most_common_class = Counter(class_values).most_common(1)[0][0]
                class_list.append(most_common_class)
            else:
                # Если нет точек в квадрате, добавляем нули
                stats_list.append([0, 0, 0, 0])
                class_list.append(0)  # Предполагаем, что если точек нет, то класс будет 0

    # Преобразуем в массивы и возвращаем
    stats_array = np.array(stats_list).reshape(len(x_bins) - 1, len(y_bins) - 1, 3)
    class_array = np.array(class_list).reshape(len(x_bins) - 1, len(y_bins) - 1, 1)
    return stats_array, class_array

# Функция обработки одного файла LAS
def process_single_las_file(file_path, output_dir, grid_size=1):
    # Загружаем LAS файл
    las = laspy.read(file_path)
    points = np.vstack([las.x, las.y, las.z]).T  # Выбираем координаты x, y, z
    classes = np.array(las.classification)  # Классы точек

    # Смещаем по оси Z так, чтобы zmin=0
    points = shift_z_to_zero(points)

    # Рассчитываем статистические параметры по квадратно-метровым блокам и классы
    stats_array, class_array = calculate_statistics_and_class(points, classes, grid_size)

    # Сохраняем результаты в .npy файлы
    file_name = os.path.basename(file_path).replace('.las', '')
    np.save(os.path.join(output_dir, f'{file_name}_x.npy'), stats_array)  # Данные x
    np.save(os.path.join(output_dir, f'{file_name}_y.npy'), class_array)  # Данные y

    print(f'Обработан файл: {file_name}')

# Функция обработки всех файлов в каталоге с параллельной обработкой
def process_all_las_files(input_dir, output_dir, grid_size=1, num_workers=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Получаем список всех файлов .las в каталоге
    las_files = [f for f in os.listdir(input_dir) if f.endswith('.las')]
    full_file_paths = [os.path.join(input_dir, f) for f in las_files]

    # Параллельная обработка файлов
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_las_file, file_path, output_dir, grid_size) for file_path in full_file_paths]

        # Ждем завершения всех задач
        for future in futures:
            future.result()

if __name__ == '__main__':
    # Пример использования
    input_directory = (r'C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing'
                       r'\data\las_org\data_las_stpls3d\LosAngeles_test_parallel')  # Путь к каталогу с файлами LAS
    output_directory = (r'C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing'
                        r'\data\las_org\data_las_stpls3d\Processed')  # Путь к каталогу для сохранения результатов
    grid_size = 0.78125  # Размер квадрата для группировки точек
    num_workers = 5  # Количество процессов для параллельной обработки

    process_all_las_files(input_directory, output_directory, grid_size, num_workers)
