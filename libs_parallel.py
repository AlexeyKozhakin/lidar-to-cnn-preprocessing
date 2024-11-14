import os
from libs import (ply_to_las_rgb,
                  get_filenames_without_extension
                  )
from multiprocessing import Pool
import subprocess
import matplotlib.pyplot as plt

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
    output_subdir = os.path.join(output_directory, os.path.splitext(filename)[0])

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