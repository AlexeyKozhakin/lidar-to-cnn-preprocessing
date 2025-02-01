from libs import (ply_to_las_rgb, generate_dataset, mask_to_las_with_class_nn_rgb,
                  generate_dataset_predict,
                  process_las_files,
                  get_filenames_without_extension
                  )
from libs_parallel import (main_parallel_ply2las,
                           main_parallel_cut_tiles,
                           generate_dataset_parallel)
import os
import time




''' Старая версия
class_colors_stpls3d = {
    0: (0, 0, 0),          # Unclassified - Черный
    1: (0, 255, 0),       # Ground - Зеленый
    2: (255, 255, 0),     # LowVegetation - Желтый
    3: (0, 0, 255),       # MediumVegetation - Синий
    4: (255, 0, 0),       # HighVegetation - Красный
    5: (0, 255, 255),     # Vehicle - Бирюзовый
    6: (255, 0, 255),     # Truck - Магента
    7: (255, 128, 0),     # Aircraft - Оранжевый
    8: (128, 128, 128),   # MilitaryVehicle - Серый
    9: (255, 20, 147),    # Bike - Deep Pink
    10: (255, 69, 0),     # Motorcycle - Красный Апельсин
    11: (210, 180, 140),  # LightPole - Бежевый
    12: (255, 105, 180),  # StreetSign - Hot Pink
    13: (165, 42, 42),    # Clutter - Коричневый
    14: (139, 69, 19),     # Fence - Темно-коричневый
    15: (128, 0, 128),    # Road - Фиолетовый
    16: (255, 255, 255),   # Windows - Белый
    17: (222, 184, 135),   # Dirt - Седой
    18: (127, 255, 0),     # Grass - Ярко-зеленый
}


0-Ground: including grass, paved road, dirt, etc.
1-Building: including commercial, residential, educational buildings.
2-LowVegetation: 0.5 m < vegetation height < 2.0 m.
3-MediumVegetation: 2.0 m < vegetation height < 5.0 m.
4-HighVegetation: 5.0 m < vegetation height.
5-Vehicle: including sedans and hatchback cars.
6-Truck: including pickup trucks, cement trucks, flat-bed trailers, trailer trucks, etc.
7-Aircraft: including helicopters and airplanes.
8-MilitaryVehicle: including tanks and Humvees.
9-Bike: bicycles.
10-Motorcycle: motorcycles.
11-LightPole: including light poles and traffic lights.
12-StreetSgin: including road signs erected at the side of roads.
13-Clutter: including city furniture, construction equipment, barricades, and other 3D shapes.
14-Fence: including timber, brick, concrete, metal fences.
15-Road: including asphalt and concrete roads.
17-Windows: glass windows.
18-Dirt: bare earth.
19-Grass: including grass lawn, wild grass, etc.

'''
'''
class_colors_stpls3d = {
    0: (0, 0, 0),          # Ground - Черный
    1: (128, 128, 128),       # Building - Зеленый           1==17
    2: (255, 255, 0),     # LowVegetation - Желтый
    3: (0, 0, 255),       # MediumVegetation - Синий
    4: (255, 0, 0),       # HighVegetation - Красный
    5: (0, 255, 255),     # Vehicle - Бирюзовый
    6: (255, 0, 255),     # Truck - Магента
    7: (255, 128, 0),     # Aircraft - Оранжевый
    8: (255, 0, 255),   # MilitaryVehicle - Серый             8==6
    9: (255, 20, 147),    # Bike - Deep Pink
    10: (255, 69, 0),     # Motorcycle - Красный Апельсин
    11: (210, 180, 140),  # LightPole - Бежевый
    12: (255, 105, 180),  # StreetSign - Hot Pink
    13: (165, 42, 42),    # Clutter - Коричневый
    14: (139, 69, 19),     # Fence - Темно-коричневый
    15: (128, 0, 128),    # Road - Фиолетовый
    17: (128, 128, 128),   # Windows - Белый
    18: (222, 184, 135),   # Dirt - Седой
    19: (127, 255, 0),     # Grass - Ярко-зеленый
}'''
# 20 - 1 (16) - 1 (17) - 1 (8) = 17 классов
class_colors_stpls3d = {
    0: (0, 0, 0),  # Ground - Черный
    1: (128, 128, 128),  # Building - Зеленый           1==17
    2: (255, 255, 0),  # LowVegetation - Желтый
    3: (0, 0, 255),  # MediumVegetation - Синий
    4: (255, 0, 0),  # HighVegetation - Красный
    5: (0, 255, 255),  # Vehicle - Бирюзовый
    6: (255, 0, 255),  # Truck - Магента
    7: (255, 128, 0),  # Aircraft - Оранжевый
    9: (255, 20, 147),  # Bike - Deep Pink
    10: (255, 69, 0),  # Motorcycle - Красный Апельсин
    11: (210, 180, 140),  # LightPole - Бежевый
    12: (255, 105, 180),  # StreetSign - Hot Pink
    13: (165, 42, 42),  # Clutter - Коричневый
    14: (139, 69, 19),  # Fence - Темно-коричневый
    15: (128, 0, 128),  # Road - Фиолетовый
    18: (222, 184, 135),  # Dirt - Седой
    19: (127, 255, 0),  # Grass - Ярко-зеленый
}



# Пример использования функции

ply_to_las_parallel = False
cut_las_parallel = True
show_stat_files_parallel = False
gen_dataset_parallel = False
gen_clouds_dataset_parallel = False

data_dir = '/mnt/working-ssd/alexey_kozhakin/MUSAC/data/STPLS3D'

if __name__ == '__main__':
    # Трансформируем оргиниальный датасет из ply в las
    if ply_to_las_parallel:
        
        ply_dir = os.path.join(data_dir,"data_org")
        las_dir =  os.path.join(data_dir,"data_las")
        start = time.time()
        main_parallel_ply2las(num_workers=67, ply_dir=ply_dir, las_dir=las_dir)
        end = time.time()
        print(end - start)

    # Нарезаем на tiles
    if cut_las_parallel:
        # Указываем путь к исходной директории с .las файлами
        input_directory = os.path.join(data_dir,"data_las")
        # Указываем путь к директории, куда будут сохраняться нарезанные файлы
        output_directory = os.path.join(data_dir,"data_las_cut_100")
        main_parallel_cut_tiles(input_directory, output_directory, tile_size=100, num_processes=90)

#========================================> Анализ нарезанных файлов <==================================================

    if show_stat_files_parallel:
        from libs_parallel import get_file_sizes_parallel, save_histogram
        import numpy as np

        # Пример использования
        directory = os.path.join(data_dir,"data_las_cut_64")
        start = time.time()
        file_sizes = get_file_sizes_parallel(directory, num_processes=90)
        end = time.time()
        print(round(end-start))
        # Вычисление статистик
        length = len(file_sizes)
        mean = np.mean(file_sizes)
        std = np.std(file_sizes)
        max_value = np.max(file_sizes)
        min_value = np.min(file_sizes)
        mean_minus_3std = mean - 3 * std

        # Вывод результатов
        print('Statistics of DataSet:')
        print(f'Files: {length}')
        print(f'Mean: {mean/1e6}')
        print(f'STD: {std/1e6}')
        print(f'Min: {min_value/1e6}')
        print(f'Max: {max_value/1e6}')
        print(f'Mean - 3*STD: {mean_minus_3std/1e6}')

        if file_sizes:
            save_histogram(file_sizes)
        else:
            print("В каталоге нет файлов или не удалось получить их размеры.")

    # ========================================> Генерирование датасета <===============================================

    if gen_dataset_parallel:
        # Пример использования
        directory = os.path.join(data_dir, 'data_las_cut_64')
        las_files = os.listdir(directory)
        las_files = [os.path.join(directory, las_file) for las_file in las_files]

        output_dir = os.path.join(data_dir,"data_training_64_512")
        start = time.time()
        generate_dataset_parallel(las_files, output_dir, class_colors_stpls3d,
                                  train_size=0.7, val_size=0.15, test_size=0.15, grid_size=512, num_processes=95)
        end = time.time()
        print(round(end-start))

#======================================= Генерация облаков точек ======================================================
    if gen_clouds_dataset_parallel:
        from libs_parallel import process_las_files_gen_clouds_parallel
        import json
        # Укажите путь к вашему config.json
        config_path = "config.json"
        """Загружает настройки из config.json."""
        with open(config_path, "r") as f:
            config = json.load(f)
            # Загружаем настройки
            points_per_cloud = config["points_per_cloud"]
            num_files = config["num_files"]
            batches_per_file = config["batches_per_file"]
            input_directory = config["input_directory"]
            output_directory = config["output_directory"]
            output_files = config["output_files"]
            num_processes = config["num_processes"]  # Количество процессов для обработки файлов
            os.makedirs(output_directory, exist_ok=True)

        full_out_dir = os.path.join(output_directory,output_files)
        # Пример вызова функции
        start = time.time()
        process_las_files_gen_clouds_parallel(input_directory, full_out_dir,
                                              num_files=num_files,
                                              num_points_lim=points_per_cloud,
                                              num_processes=num_processes)
        end = time.time()
        print(f'Обработка завершена. Время: {end - start:.2f} секунд')






# if gen_predict_dataset:
#     # path_to_las = r"C:\Users\alexe\Downloads\UM\work\Data\2012-20240803T112213Z-001\2018-20240913T181437Z-001\city\city"
#     path_to_las = (r"D:\data\las_org\san_gwan_256_256_1")
#     # Ищем все файлы с расширением .las в указанном каталоге
#     las_files = glob.glob(os.path.join(path_to_las, '*.las'))
#
#     output_dir = r"D:\data\data_for_training\data_training_stpl3d_256_2048"
#     generate_dataset_predict(las_files, output_dir, dataset_type='predict', grid_size=2048, mask=False)
#
# if gen_colored_las:
#     # Пример использования
#     # city
#     # las_directory = 'data\las_org\data_las_musac_2018_city'
#     # mask_directory = 'data\predicted\stpls3d_tenser_128_city\predict_2018_stpls3d_ciy_epoch_2'
#     # output_directory = 'data\las_colored\stpls3d_tenser_128\data_las_musac_2018_colored_500x500_city'
#     # countryside
#     las_directory = 'D:\data\las_org\san_gwan_256_256_1'
#     mask_directory = 'D:\data\data_for_training\data_training_stpl3d_256\predict\segment_pytorch'
#     output_directory = 'D:\data\san_gwann_colored_256x256_pytorch'
#     filenames = get_filenames_without_extension(las_directory)
#     for file in filenames:
#         las_file_path = os.path.join(las_directory, file + '.las')
#         mask_file_path = os.path.join(mask_directory, file + '.png')
#         output_las_path = os.path.join(output_directory, file + '.las')
#         print(las_file_path)
#         print(mask_file_path)
#         print(output_las_path)
#         mask_to_las_with_class_nn_rgb(las_file_path, mask_file_path, output_las_path,
#                                       class_colors_stpls3d,
#                                       grid_size=500)
# # Очистка выбросов по Z
# if clean_dataset:
#     # Укажите путь к каталогу с LAS-файлами
#     # directory = (r"C:\Users\alexe\Downloads\UM\work\Data"
#     #              r"\2012-20240803T112213Z-001\2018-20240913T181437Z-001\city\city")
#     # directory = (r"C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing"
#     # r"\data\las_org"
#     # r"\data_las_musac_2018_san_gwan")
#     directory = (r"C:\Users\alexe\Downloads\UM\work\Data\2012-20240803T112213Z-001\2018-20240913T181437Z-001"
#                  r"\city_san_gwann"
#                  r"\san_gwan_256_256_1")
#     process_las_files(directory)
#

