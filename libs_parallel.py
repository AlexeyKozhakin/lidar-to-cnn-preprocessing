import os
from libs import (ply_to_las_rgb,
                  get_filenames_without_extension
                  )

import subprocess
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import random
from multiprocessing import Pool
import laspy
#==========================================> ply2las_rgb <============================================================

def process_file_ply2las_rgb(file_data):
    """
    Обработка одного файла PLY и его сохранение как LAS.
    """
    file, ply_dir, las_dir = file_data
    print(f'Now processing {file}')
    ply_file = os.path.join(ply_dir, file + '.ply')
    las_file = os.path.join(las_dir, file + '.las')
    ply_to_las_rgb(ply_file, las_file, dataset='stpls3d')


def main_parallel_ply2las(num_workers = 1, ply_dir='', las_dir=''):
    filenames = get_filenames_without_extension(ply_dir)

    # Создание списка аргументов для передачи каждому процессу
    file_data_list = [(file, ply_dir, las_dir) for file in filenames]

    # Параллельная обработка с использованием пула процессов
    with Pool(processes=num_workers) as pool:
        pool.map(process_file_ply2las_rgb, file_data_list)

#==========================================> ply2las_rgb <============================================================

def process_file_cut_tiles(filename, input_directory, output_directory, tile_size=64):
    """
    Функция для нарезки одного файла .las с помощью lastile.
    """
    input_file = os.path.join(input_directory, filename)
    output_subdir = os.path.join(output_directory, filename.split('.las')[0])

    # Создаем подкаталог для текущего файла, если его нет
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    # Формируем команду для lastile
    command = [
        'lastile',
        '-i', input_file,
        '-tile_size', str(tile_size),
        '-o', output_subdir,
    ]

    # Выполняем команду
    subprocess.run(command)
    print(f"{filename} успешно нарезан и сохранен в {output_subdir}")

def main_parallel_cut_tiles(input_directory, output_directory, tile_size=64, num_processes=1):
    # Создаем выходную директорию, если ее не существует
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Получаем список файлов .las в исходной директории
    filenames = [f for f in os.listdir(input_directory) if f.endswith('.las')]

    # Используем Pool для параллельной обработки файлов
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_file_cut_tiles, [(filename, input_directory, output_directory, tile_size) for filename in filenames])

    print("Все файлы успешно нарезаны и сохранены.")

#====================================> Проверка размеров датасета, статистика и удаление <============================
def get_file_size(filepath):
    """Получает размер файла, если доступен."""
    try:
        return os.path.getsize(filepath)
    except OSError:
        print(f"Не удалось получить размер файла: {filepath}")
        return 0  # Возвращаем 0, если размер не удалось получить


def get_file_sizes_parallel(directory, num_processes=None):
    """Параллельно вычисляет размеры всех файлов в указанной директории."""
    # Собираем список всех файлов в директории и поддиректориях
    filepaths = []
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            filepaths.append(os.path.join(foldername, filename))

    # Используем Pool для параллельного выполнения get_file_size на каждом файле
    with Pool(processes=num_processes) as pool:
        file_sizes = pool.map(get_file_size, filepaths)

    return file_sizes

def save_histogram(file_sizes, bin_size=50, output_path="histogram.png"):
    plt.figure(figsize=(10, 6))
    # Строим гистограмму по размерам файлов
    plt.hist(file_sizes, bins=bin_size, edgecolor='black')
    plt.title('Histogram of File Sizes')
    plt.xlabel('File Size (bytes)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Сохраняем график в файл
    plt.savefig(output_path, format='png')
    plt.close()  # Закрываем, чтобы освободить память

    print(f"Гистограмма сохранена в файл: {output_path}")

#====================================> Генерация датасета <=============================================
from sklearn.model_selection import train_test_split
from libs import generate_data_from_las, create_directory
def process_file_gen_data(file_path, original_dir, segment_dir, class_colors, grid_size):
    # Генерация имени файла
    file_name = os.path.basename(file_path).replace('.las', '.png')

    # Полные пути для сохранения оригинала и маски
    original_output_path = os.path.join(original_dir, file_name)
    segment_output_path = os.path.join(segment_dir, file_name)

    # Генерация данных
    generate_data_from_las(file_path, original_output_path, segment_output_path,
                           class_colors, grid_size, mask=True)

    print(f"Сгенерировано изображение и маска для {file_name}")

def generate_dataset_parallel(las_files, output_dir, class_colors,
                     train_size=0.7, val_size=0.15, test_size=0.15, grid_size=500, num_processes=None):
    # Проверка, что размеры датасетов в сумме дают 1
    assert train_size + val_size + test_size == 1, "Train, val and test sizes should sum to 1."

    # Разделение файлов на train, val и test
    train_files, temp_files = train_test_split(las_files, test_size=(1 - train_size), random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=(test_size / (val_size + test_size)), random_state=42)

    # Структура директорий
    dataset_structure = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for dataset_type, files in dataset_structure.items():
        original_dir = os.path.join(output_dir, dataset_type, 'original')
        segment_dir = os.path.join(output_dir, dataset_type, 'segment')

        # Создаем директории, если они не существуют
        create_directory(original_dir)
        create_directory(segment_dir)

        # Подготавливаем аргументы для каждого файла
        args = [(file_path, original_dir, segment_dir, class_colors, grid_size) for file_path in files]

        # Запуск в параллельном режиме с использованием пула процессов
        with Pool(processes=num_processes) as pool:
            pool.starmap(process_file_gen_data, args)

    print("Датасет успешно сгенерирован.")


#======================================== Генерация облаков точек в параллельном режиме ===============================
def count_points_in_las(file_path):
    """Возвращает количество точек в LAS файле."""
    las = laspy.read(file_path)
    return len(las.points)


def random_point_sampling(points, n_samples):
    """Случайно выбирает n_samples точек из облака."""
    if len(points) <= n_samples:
        return points  # Если точек меньше или равно n_samples, возвращаем все точки

    indices = np.random.choice(len(points), n_samples, replace=False)  # Случайные индексы без замены
    return points[indices]


def process_las_file(file_path, num_points_lim):
    """Обрабатывает один LAS файл и возвращает выборку точек и классов."""
    try:
        num_points = count_points_in_las(file_path)
        #print(num_points_lim)

        if num_points > num_points_lim:
            las = laspy.read(file_path)
            #points = np.vstack((las.x, las.y, las.z)).T  # Формируем массив точек (N, 3)
            points = np.vstack((las.x, las.y, las.z, las.red, las.green, las.blue)).T
            classes = las.classification  # Извлечение классов точек

            # Случайно выбираем 4096 точек
            sampled_points = random_point_sampling(points, num_points_lim)

            # Получаем классы для отобранных точек
            sampled_indices = np.random.choice(num_points, num_points_lim, replace=False)
            sampled_classes = classes[sampled_indices]  # Получаем классы для выбранных точек

            # Объединяем координаты и классы
            return np.hstack((sampled_points, sampled_classes[:, np.newaxis]))
        else:
            return None
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return None


def process_las_files_gen_clouds_parallel(directory, output_path, num_points_lim=4096, num_files=1,
                                          num_processes=1):
    """
    Считывает LAS файлы и записывает данные в несколько .pt файлов, записывая их по частям, чтобы экономить память.

    Параметры:
    directory (str): Путь к директории с LAS файлами.
    output_path (str): Базовый путь для сохранения файлов .pt.
    num_points_lim (int): Лимит на количество точек в каждом файле (по умолчанию 4096).
    num_files (int): Количество файлов для разделения (по умолчанию 1).
    num_processes (int): Количество процессов для параллельной обработки (по умолчанию количество ядер CPU).
    """
    las_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.las')]

    # Используем Pool для параллельной обработки файлов
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_las_file, [(file_path, num_points_lim) for file_path in las_files])

    # Фильтруем результаты, чтобы убрать None
    results = [result for result in results if result is not None]


    if not results:
        print(f"Нет файлов с количеством точек больше {num_points_lim}.")
        return

    # Разделяем результаты на части и сохраняем в файлы
    num_results = len(results)
    results_per_file = num_results // num_files

    for i in range(num_files):
        start_idx = i * results_per_file
        end_idx = start_idx + results_per_file if i < num_files - 1 else num_results

        chunk_results = results[start_idx:end_idx]
        chunk_output_path = f"{output_path}_part{i + 1}.pt"
        print(torch.tensor(np.array(chunk_results)).shape)
        # Преобразуем в PyTorch Tensor и сохраняем
        torch.save(torch.tensor(np.array(chunk_results)), chunk_output_path)
        print(f"Часть {i + 1} сохранена в {chunk_output_path}.")