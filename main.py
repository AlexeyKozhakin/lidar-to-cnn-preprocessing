from libs import (ply_to_las, generate_dataset, mask_to_las_with_class_nn_rgb,
                  generate_dataset_predict,
                  process_las_files,
                  get_filenames_without_extension
                  )
import os
import glob
import subprocess




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


class_colors_toronto3D_org = {
    0: (0, 0, 0),  # Unclassified - Черный
    1: (0, 255, 0),  # Ground - Зеленый
    2: (255, 255, 0),  # Road_markings - Желтый
    3: (0, 0, 255),  # Natural - Синий
    4: (255, 0, 0),  # Building - Красный
    5: (0, 255, 255),  # Utility_line - Бирюзовый
    6: (255, 0, 255),  # Pole - Магента
    7: (255, 128, 0),  # Car - Оранжевый
    8: (128, 128, 128),  # Fence - Серый
}



class_colors_hessinheim = {
    0: (34, 139, 34),   # Low Vegetation - Низкая растительность
    1: (128, 128, 128), # Impervious Surface - Непроницаемая поверхность
    2: (255, 0, 0),     # Vehicle - Транспортное средство
    3: (255, 165, 0),   # Urban Furniture - Городская мебель
    4: (0, 0, 255),     # Roof - Крыша
    5: (128, 0, 128),   # Façade - Фасад
    6: (0, 255, 0),     # Shrub - Кустарник
    7: (0, 100, 0),     # Tree - Дерево
    8: (139, 69, 19),   # Soil/Gravel - Почва/Гравий
    9: (64, 224, 208),  # Vertical Surface - Вертикальная поверхность
    10: (255, 255, 0),  # Chimney - Дымоход
}


# Пример использования функции
ply_to_las_processing = False
cut_las = False
gen_dataset = False
gen_predict_dataset = False
gen_colored_las = True
clean_dataset = False #Убрать выбросы по Z


# if ply_to_las_processing:
#     ply_file = r"C:\Users\alexe\Downloads\UM\work\Data" \
#                 r"\STPLS3D_ply\STPLS3D_ply\STPLS3D\Synthetic_v1\LosAngeles.ply"
#     las_file = r"C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing" \
#                 r"\data\las_org\data_las_stpls3d\LosAngeles.las"
#     ply_to_las(ply_file, las_file, dataset = 'stpls3d')


if ply_to_las_processing:
    ply_dir = r"C:\Users\alexe\Downloads\UM\work\Data" \
              r"\STPLS3D_ply\STPLS3D_ply\STPLS3D\RealWorldData"
                # r"\STPLS3D_ply\STPLS3D_ply\STPLS3D\Synthetic_v3"
    # las_dir = r"C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing" \
    #             r"\data\las_org\data_las_stpls3d\all_org_las"
    las_dir = r"D:\data\las_org\data_las_stpls3d\all_org_las"
    filenames = get_filenames_without_extension(ply_dir)
    for file in filenames:
    #for file in ['427250_3989900']:
        print(f'Now processing {file}')
        ply_file = os.path.join(ply_dir, file + '.ply')
        las_file = os.path.join(las_dir, file+'.las')
        ply_to_las(ply_file, las_file, dataset='stpls3d')

if cut_las:

    # Указываем путь к исходной директории с .las файлами
    input_directory = r"D:\data\las_org\data_las_stpls3d\all_org_las"

    # Указываем путь к директории, куда будут сохраняться нарезанные файлы
    output_directory = r"D:\data\las_org\data_las_stpls3d\all_org_las_cut_50"

    # Создаем выходную директорию, если она не существует
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Размер тайла
    tile_size = 50

    # Проходим по всем файлам в исходной директории
    filenames = os.listdir(input_directory)
    for filename in filenames[2:]:
        if filename.endswith('.las'):
            input_file = os.path.join(input_directory, filename)

            # Указываем имя для выходного каталога для каждого файла
            output_subdir = os.path.join(output_directory, os.path.splitext(filename)[0])

            # Создаем подкаталог для каждого файла
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            # Формируем команду для lastile
            command = [
                'lastile',
                '-i', input_file,
                '-tile_size', str(tile_size),
                '-o', output_subdir,  # Указываем, куда записывать нарезанные файлы
            ]

            # Выполняем команду
            subprocess.run(command)

    print("Все файлы успешно нарезаны и сохранены.")


if gen_dataset:
    # Пример использования
    directory = r'D:\data\las_org\data_las_stpls3d\all_org_las_cut_50'
    las_files = os.listdir(directory)
    las_files = [os.path.join(directory,las_file) for las_file in las_files]

    output_dir = r"D:\data\data_for_training\data_training_stpl3d"
    generate_dataset(las_files, output_dir,
                     class_colors_stpls3d,
                     train_size=0.7, val_size=0.15, test_size=0.15, grid_size=500)



if gen_predict_dataset:
    #path_to_las = r"C:\Users\alexe\Downloads\UM\work\Data\2012-20240803T112213Z-001\2018-20240913T181437Z-001\city\city"
    path_to_las =  (r"C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing"
    r"\data\las_org"
    r"\san_gwan_50x50")
    # Ищем все файлы с расширением .las в указанном каталоге
    las_files = glob.glob(os.path.join(path_to_las, '*.las'))

    output_dir = r"data_img\MUSAC-UNET\musac_only_2018_san_gwan"
    generate_dataset_predict(las_files, output_dir, dataset_type='test', mask=False)



if gen_colored_las:
    # Пример использования
    #city
    # las_directory = 'data\las_org\data_las_musac_2018_city'
    # mask_directory = 'data\predicted\stpls3d_tenser_128_city\predict_2018_stpls3d_ciy_epoch_2'
    # output_directory = 'data\las_colored\stpls3d_tenser_128\data_las_musac_2018_colored_500x500_city'
    #countryside
    las_directory = 'data\las_org\san_gwan_50x50'
    mask_directory = 'data\predicted\stpls3d_tenser_128_san_gwan\predict_2018_stpls3d_san_gwan'
    output_directory = 'data\las_colored\stpls3d_tenser_128\data_las_musac_2018_colored_500x500_san_gwan'
    filenames = get_filenames_without_extension(las_directory)
    for file in filenames:
        las_file_path = os.path.join(las_directory, file+'.las')
        mask_file_path = os.path.join(mask_directory, file+'.png')
        output_las_path = os.path.join(output_directory, file+'.las')
        print(las_file_path)
        print(mask_file_path)
        print(output_las_path)
        mask_to_las_with_class_nn_rgb(las_file_path, mask_file_path, output_las_path,
                                  class_colors_stpls3d,
                                   grid_size=500)
#Очистка выбросов по Z
if clean_dataset:
    # Укажите путь к каталогу с LAS-файлами
    # directory = (r"C:\Users\alexe\Downloads\UM\work\Data"
    #              r"\2012-20240803T112213Z-001\2018-20240913T181437Z-001\city\city")
    # directory = (r"C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing"
    # r"\data\las_org"
    # r"\data_las_musac_2018_san_gwan")
    directory = (r"C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing"
    r"\data\las_org"
    r"\san_gwan_50x50")
    process_las_files(directory)