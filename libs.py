from plyfile import PlyData
import laspy
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PIL import Image
import open3d as o3d



def ply_to_las(ply_file_path, las_file_path, dataset = 'toronto3d'):
    """
    Конвертирует PLY файл в LAS файл с координатами и метками классификации.

    :param ply_file_path: Путь к PLY файлу.
    :param las_file_path: Путь для сохранения выходного LAS файла.
    """
    # Открываем PLY файл
    ply_data = PlyData.read(ply_file_path)

    # Доступ к элементам vertex
    vertex_data = ply_data['vertex'].data
    # Проверяем тип и структуру данных
    print(vertex_data.dtype)
    print(vertex_data['class'][:10])
#    print(vertex_data['instance'][:10])
    # Извлекаем координаты x, y, z и метки scalar_Label
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    if dataset == 'toronto3d':
        labels = vertex_data['scalar_Label']  # Метки классификации
    elif dataset == 'stpls3d':
        labels = vertex_data['class']
    # Создаем массив с координатами точек (x, y, z)
    points = np.vstack([x, y, z]).T

    # Создание LAS файла с версией 1.2 и форматом точек 3
    las_file = laspy.create(file_version="1.2", point_format=3)

    # Установка координат в LAS файл
    las_file.x = points[:, 0]
    las_file.y = points[:, 1]
    las_file.z = points[:, 2]

    # Установка меток классификации в LAS файл
    las_file.classification = labels.astype(np.uint8)  # Преобразуем метки в uint8

    # Сохранение LAS файла
    las_file.write(las_file_path)

    print(f"Файл {las_file_path} успешно создан.")



# Функция для создания директорий, если их нет
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Функция для генерации данных (используем ранее написанные функции)
def generate_data_from_las(file_path, original_output_path, segment_output_path,
                           class_colors=False,
                           grid_size=500, mask=False):
    # Генерация оригинального изображения (высота)
    las_to_image(file_path, original_output_path, grid_size)

    if mask:
    # Генерация маски сегментации
        las_to_class_mask(file_path, segment_output_path, class_colors, grid_size)

# Главная функция для генерации базы данных
def generate_dataset(las_files, output_dir,
                     class_colors,
                     train_size=0.7, val_size=0.15, test_size=0.15, grid_size=500):
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

        for file_path in files:
            # Генерация имен файлов
            file_name = os.path.basename(file_path).replace('.las', '.png')

            # Полные пути для сохранения оригинала и маски
            original_output_path = os.path.join(original_dir, file_name)
            segment_output_path = os.path.join(segment_dir, file_name)

            # Генерация данных
            generate_data_from_las(file_path, original_output_path, segment_output_path,
                                   class_colors, grid_size, mask=True)

            print(f"Сгенерировано изображение и маска для {file_name}")


def generate_dataset_predict(las_files, output_dir, dataset_type='test', grid_size=512, mask=False):


        original_dir = os.path.join(output_dir, dataset_type, 'original')
        segment_dir = os.path.join(output_dir, dataset_type, 'segment')

        # Создаем директории, если они не существуют
        create_directory(original_dir)
        create_directory(segment_dir)

        for file_path in las_files:
            # Генерация имен файлов
            file_name = os.path.basename(file_path).replace('.las', '.png')

            # Полные пути для сохранения оригинала и маски
            original_output_path = os.path.join(original_dir, file_name)
            segment_output_path = os.path.join(segment_dir, file_name)

            # Генерация данных
            generate_data_from_las(file_path, original_output_path, segment_output_path, grid_size=grid_size, mask=False)

            print(f"Сгенерировано изображение и маска для {file_name}")



def las_to_image(file_path, output_image_path, grid_size=500):
    # Загрузка файла .las
    las = laspy.read(file_path)

    # Извлечение координат
    x, y, z = las.x, las.y, las.z

    # Смещение координат, чтобы они начинались с 0
    x_min, y_min = np.min(x), np.min(y)
    x_shifted = x - x_min
    y_shifted = y - y_min

    # Создание сетки grid_size на grid_size
    xi = np.linspace(0, np.max(x_shifted), grid_size)
    yi = np.linspace(0, np.max(y_shifted), grid_size)
    xi, yi = np.meshgrid(xi, yi)

    # Интерполяция z-значений на сетке с использованием ближайших соседей
    zi = griddata((x_shifted, y_shifted), z, (xi, yi), method='nearest')

    # Нормализация значений z для преобразования их в диапазон от 0 до 255
    zi_normalized = (zi - np.min(zi)) / (np.max(zi) - np.min(zi))  # Нормализация в диапазон [0, 1]
    zi_scaled = (zi_normalized * 255).astype(np.uint8)  # Преобразование в диапазон [0, 255]

    # Создание трехканального изображения, где каждый канал содержит одинаковые данные z
    rgb_image = np.stack([zi_scaled, zi_scaled, zi_scaled], axis=-1)

    # Сохранение изображения в файл
    img = Image.fromarray(rgb_image)
    img.save(output_image_path)

    print(f"Изображение сохранено по пути: {output_image_path}")


def compute_curvatures(z, x, y):
    """ Вспомогательная функция для вычисления кривизны k1 и k2 из высотной карты. """
    # Применение градиента для нахождения производных
    dzdx, dzdy = np.gradient(z)  # Используем градиенты по умолчанию

    # Вычисление второй производной для кривизны
    d2zdx2 = np.gradient(dzdx, axis=0)
    d2zdy2 = np.gradient(dzdy, axis=1)
    d2zdxdy = np.gradient(dzdx, axis=1)  # Смешанная производная

    # Вычисление кривизны
    k1 = d2zdx2 + d2zdy2  # Можно изменить на k1 = d2zdx2
    k2 = d2zdx2 - d2zdy2  # Здесь также можно изменить в зависимости от требований

    return k1, k2

def las_to_image_with_curve(file_path, output_image_path, grid_size=500):
    # Загрузка файла .las
    las = laspy.read(file_path)

    # Извлечение координат
    x, y, z = las.x, las.y, las.z

    # Смещение координат, чтобы они начинались с 0
    x_min, y_min = np.min(x), np.min(y)
    x_shifted = x - x_min
    y_shifted = y - y_min

    # Создание сетки grid_size на grid_size
    xi = np.linspace(0, np.max(x_shifted), grid_size)
    yi = np.linspace(0, np.max(y_shifted), grid_size)
    xi, yi = np.meshgrid(xi, yi)

    # Интерполяция z-значений на сетке с использованием ближайших соседей
    zi = griddata((x_shifted, y_shifted), z, (xi, yi), method='nearest')

    # Вычисление кривизны
    k1, k2 = compute_curvatures(zi, xi, yi)

    # Нормализация значений z для преобразования их в диапазон от 0 до 255
    zi_normalized = (zi - np.min(zi)) / (np.max(zi) - np.min(zi))  # Нормализация в диапазон [0, 1]
    zi_scaled = (zi_normalized * 255).astype(np.uint8)  # Преобразование в диапазон [0, 255]

    # Нормализация кривизны k1 и k2
    k1_normalized = (k1 - np.min(k1)) / (np.max(k1) - np.min(k1))  # Нормализация в диапазон [0, 1]
    k2_normalized = (k2 - np.min(k2)) / (np.max(k2) - np.min(k2))  # Нормализация в диапазон [0, 1]

    # Преобразование в диапазон [0, 255]
    k1_scaled = (k1_normalized * 255).astype(np.uint8)
    k2_scaled = (k2_normalized * 255).astype(np.uint8)

    # Создание трехканального изображения
    rgb_image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    rgb_image[..., 0] = zi_scaled  # z в первом канале
    rgb_image[..., 1] = k1_scaled   # k1 во втором канале
    rgb_image[..., 2] = k2_scaled   # k2 в третьем канале

    # Сохранение изображения в файл
    img = Image.fromarray(rgb_image)
    img.save(output_image_path)

    print(f"Изображение сохранено по пути: {output_image_path}")



def las_to_class_mask(file_path, output_image_path, class_colors, grid_size=500):
    # Загрузка файла .las
    las = laspy.read(file_path)

    # Извлечение координат и классов
    x, y, z = las.x, las.y, las.z
    classes = las.classification  # Извлечение классов точек

    # Смещение координат, чтобы они начинались с 0
    x_min, y_min = np.min(x), np.min(y)
    x_shifted = x - x_min
    y_shifted = y - y_min

    # Создание сетки grid_size на grid_size
    xi = np.linspace(0, np.max(x_shifted), grid_size)
    yi = np.linspace(0, np.max(y_shifted), grid_size)
    xi, yi = np.meshgrid(xi, yi)

    # Интерполяция классов на сетке с использованием ближайших соседей
    class_map = griddata((x_shifted, y_shifted), classes, (xi, yi), method='nearest')

    # Создание цветной карты
    color_image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

    # Заполнение цветами для каждого класса
    for class_id, color in class_colors.items():
        color_image[class_map == class_id] = color

    # Сохранение изображения в файл
    img = Image.fromarray(color_image)
    img.save(output_image_path)

    # Возвращаем маску
    return color_image


def mask_to_las_with_class_nn_rgb(las_file_path, image_file_path, output_las_path,
                              class_colors,
                              grid_size=500):
    # Шаг 1. Чтение исходного LAS файла
    las = laspy.read(las_file_path)

    # Извлечение координат X, Y, Z
    x, y, z = las.x, las.y, las.z

    # Шаг 2. Чтение изображения маски
    image = Image.open(image_file_path)
    image = np.array(image)

    # Предположим, что изображение имеет размер (grid_size, grid_size, 3)
    img_height, img_width, _ = image.shape

    # Шаг 3. Сдвигаем координаты X, Y так, чтобы они начинались с 0
    x_min, y_min = np.min(x), np.min(y)
    x_shifted = x - x_min
    y_shifted = y - y_min

    # Масштабируем координаты LAS файла к диапазону от 0 до 1 (нормализация)
    x_scaled = x_shifted / np.max(x_shifted)
    y_scaled = y_shifted / np.max(y_shifted)

    # Шаг 4. Преобразуем индексы пикселей изображения в координаты от 0 до 1
    xi = np.linspace(0, 1, img_width)
    yi = np.linspace(0, 1, img_height)
    xi, yi = np.meshgrid(xi, yi)

    # Преобразуем координаты сетки изображения и соответствующие цвета в 1D массивы для интерполяции
    xi_flat = xi.ravel()
    yi_flat = yi.ravel()
    colors_flat = image.reshape(-1, 3)  # Преобразуем цвета изображения в плоский массив

    # Шаг 5. Создаем интерполятор на основе ближайших соседей
    interpolator = NearestNDInterpolator(np.column_stack((xi_flat, yi_flat)), colors_flat)

    # Шаг 6. Применяем интерполяцию для каждой точки из LAS файла
    nearest_colors = interpolator(x_scaled, y_scaled)

    # Преобразуем карту цветов классов в более удобную для поиска структуру
    color_to_class = {tuple(v): k for k, v in class_colors.items()}

    # Шаг 8. Определяем класс и RGB для каждой точки на основании ближайшего цвета
    classifications = np.zeros(len(nearest_colors), dtype=np.uint8)
    rgb_values = np.zeros((len(nearest_colors), 3), dtype=np.uint16)  # для RGB значений

    for i, color in enumerate(nearest_colors):
        # Приведение цветов к целым числам для сопоставления
        color = tuple(np.round(color).astype(int))
        classifications[i] = color_to_class.get(color, 0)  # Класс по умолчанию 0 (Unclassified)

        # Записываем RGB-значения для текущего класса
        if color in color_to_class:
            rgb = np.array(color) * 256  # Преобразуем цвета в 16-битное значение для LAS
            rgb_values[i] = rgb.astype(np.uint16)
        else:
            rgb_values[i] = (0, 0, 0)  # Если класс не найден, ставим черный цвет

    # Шаг 9. Создание нового LAS файла с нужными данными (x, y, z, classification, rgb)
    new_las = laspy.create(point_format=las.point_format, file_version=las.header.version)

    # Переносим x, y, z, classification
    new_las.x = x
    new_las.y = y
    new_las.z = z
    new_las.classification = classifications

    # Проверим, поддерживает ли исходный файл LAS сохранение RGB
    if 'red' in new_las.point_format.dimension_names:
        new_las.red = rgb_values[:, 0]  # Записываем красный канал
        new_las.green = rgb_values[:, 1]  # Записываем зеленый канал
        new_las.blue = rgb_values[:, 2]  # Записываем синий канал
    else:
        # Добавим RGB каналы, если они отсутствуют
        new_las.point_format.add_extra_dimension(name='red', dtype=np.uint16)
        new_las.point_format.add_extra_dimension(name='green', dtype=np.uint16)
        new_las.point_format.add_extra_dimension(name='blue', dtype=np.uint16)

        new_las.red = rgb_values[:, 0]
        new_las.green = rgb_values[:, 1]
        new_las.blue = rgb_values[:, 2]

    # Шаг 10. Сохранение обновленного LAS файла
    new_las.write(output_las_path)

    print(f'Файл {output_las_path} успешно создан с классами точек и RGB значениями.')
def mask_to_las_with_class_nn(las_file_path, image_file_path, output_las_path,
                              class_colors,
                              grid_size=500):
    # Шаг 1. Чтение исходного LAS файла
    las = laspy.read(las_file_path)

    # Извлечение координат X, Y, Z
    x, y, z = las.x, las.y, las.z

    # Шаг 2. Чтение изображения маски
    image = Image.open(image_file_path)
    image = np.array(image)

    # Предположим, что изображение имеет размер (grid_size, grid_size, 3)
    img_height, img_width, _ = image.shape

    # Шаг 3. Сдвигаем координаты X, Y так, чтобы они начинались с 0
    x_min, y_min = np.min(x), np.min(y)
    x_shifted = x - x_min
    y_shifted = y - y_min

    # Масштабируем координаты LAS файла к диапазону от 0 до 1 (нормализация)
    x_scaled = x_shifted / np.max(x_shifted)
    y_scaled = y_shifted / np.max(y_shifted)

    # Шаг 4. Преобразуем индексы пикселей изображения в координаты от 0 до 1
    xi = np.linspace(0, 1, img_width)
    yi = np.linspace(0, 1, img_height)
    xi, yi = np.meshgrid(xi, yi)

    # Преобразуем координаты сетки изображения и соответствующие цвета в 1D массивы для интерполяции
    xi_flat = xi.ravel()
    yi_flat = yi.ravel()
    colors_flat = image.reshape(-1, 3)  # Преобразуем цвета изображения в плоский массив

    # Шаг 5. Создаем интерполятор на основе ближайших соседей
    interpolator = NearestNDInterpolator(np.column_stack((xi_flat, yi_flat)), colors_flat)

    # Шаг 6. Применяем интерполяцию для каждой точки из LAS файла
    nearest_colors = interpolator(x_scaled, y_scaled)


    # Преобразуем карту цветов классов в более удобную для поиска структуру
    color_to_class = {tuple(v): k for k, v in class_colors.items()}

    # Шаг 8. Определяем класс для каждой точки на основании ближайшего цвета
    classifications = np.zeros(len(nearest_colors), dtype=np.uint8)

    for i, color in enumerate(nearest_colors):
        # Приведение цветов к целым числам для сопоставления
        color = tuple(np.round(color).astype(int))
        classifications[i] = color_to_class.get(color, 0)  # Класс по умолчанию 0 (Unclassified)

    # Шаг 9. Создание нового LAS файла с только нужными данными
    # Используем точные данные LAS (версия и формат) для создания нового файла
    new_las = laspy.create(point_format=las.point_format, file_version=las.header.version)

    # Переносим только x, y, z и classification
    new_las.x = x
    new_las.y = y
    new_las.z = z
    new_las.classification = classifications

    # Шаг 10. Сохранение обновленного LAS файла
    new_las.write(output_las_path)

    print(f'Файл {output_las_path} успешно создан с классами точек.')


def filter_las_file(filepath):
    # Шаг 1: Чтение LAS-файла
    las_file = laspy.read(filepath)

    # Шаг 2: Извлечение координат Z (высоты)
    z_coordinates = las_file.z

    # Шаг 3: Вычисление среднего и стандартного отклонения
    mean_z = np.mean(z_coordinates)
    std_z = np.std(z_coordinates)

    # Шаг 4: Фильтрация точек на основе 3 сигм
    mask = (z_coordinates >= mean_z - 3 * std_z) & (z_coordinates <= mean_z + 3 * std_z)

    # Применение маски к остальным координатам
    las_file.points = las_file.points[mask]

    # Шаг 5: Перезапись очищенного файла под тем же именем
    las_file.write(filepath)
    print(f"Файл {filepath} успешно очищен и перезаписан.")


def process_las_files(directory):
    # Проход по всем файлам в директории
    for filename in os.listdir(directory):
        if filename.endswith(".las"):
            filepath = os.path.join(directory, filename)
            print(f"Обработка файла: {filepath}")
            filter_las_file(filepath)


def get_filenames_without_extension(directory):
    # Получаем список всех файлов в указанной директории
    filenames = os.listdir(directory)

    # Удаляем расширение у каждого файла
    filenames_without_extension = [os.path.splitext(filename)[0] for filename in filenames]

    return filenames_without_extension


def get_file_sizes(directory):
    file_sizes = []
    # Проходим по каждому файлу и подкаталогу
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(foldername, filename)
            try:
                # Получаем размер файла в байтах и добавляем в список
                file_size = os.path.getsize(filepath)
                file_sizes.append(file_size)
            except OSError:
                # Если файл не доступен или возникла ошибка доступа
                print(f"Не удалось получить размер файла: {filepath}")
    return file_sizes

def plot_histogram(file_sizes, bin_size=50):
    plt.figure(figsize=(10, 6))
    # Строим гистограмму по размерам файлов
    plt.hist(file_sizes, bins=bin_size, edgecolor='black')
    plt.title('Histogram of File Sizes')
    plt.xlabel('File Size (bytes)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    

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

def process_las_files_gen_clouds(directory, output_path, num_points_lim = 4096):
    """Считывает LAS файлы и записывает данные в .npy файл."""
    data_list = []

    las_files = [f for f in os.listdir(directory) if f.endswith('.las')]
    
    num_file = 0
    for las_file in las_files:
        num_file+=1
        if num_file%100==0:
            print(num_file)
        file_path = os.path.join(directory, las_file)
        try:
            num_points = count_points_in_las(file_path)

            if num_points > num_points_lim:
                las = laspy.read(file_path)
                points = np.vstack((las.x, las.y, las.z)).T  # Формируем массив точек (N, 3)
                classes = las.classification  # Извлечение классов точек

                # Случайно выбираем 4096 точек
                sampled_points = random_point_sampling(points, num_points_lim)
                
                # Получаем классы для отобранных точек
                sampled_indices = np.random.choice(num_points, num_points_lim, replace=False)  # Получаем индексы для классов
                sampled_classes = classes[sampled_indices]  # Получаем классы для выбранных точек

                # Объединяем координаты и классы
                data_list.append(np.hstack((sampled_points, sampled_classes[:, np.newaxis])))  # Добавляем класс
        except:
            print(f'file {las_file} возникла ошибка')
    # Преобразуем в массив NumPy
    if data_list:
        final_data = np.array(data_list)
        np.save(output_path, final_data)  # Сохраняем в .npy файл
        print(f"Данные успешно сохранены в {output_path}.")
    else:
        print(f"Нет файлов с количеством точек больше {num_points_lim}.")
        


def process_las_files_gen_clouds_by_batch(directory, output_path, num_points_lim=4096, num_files=1):
    """
    Считывает LAS файлы и записывает данные в несколько .npy файлов, записывая их по частям, чтобы экономить память.
    
    Параметры:
    directory (str): Путь к директории с LAS файлами.
    output_path (str): Базовый путь для сохранения файлов .npy.
    num_points_lim (int): Лимит на количество точек в каждом файле (по умолчанию 4096).
    num_files (int): Количество файлов для разделения (по умолчанию 1).
    """
    data_list = []
    las_files = [f for f in os.listdir(directory) if f.endswith('.las')]
    
    num_file = 0
    total_points = 0
    points_per_file = 0

    # Определяем количество файло в которые будут записаны точки
    target_points_per_file = len(las_files) // num_files
    file_counter = 1

    for las_file in las_files:
        num_file += 1
        if num_file % 100 == 0:
            print(f"Обработано {num_file} файлов")

        file_path = os.path.join(directory, las_file)
        try:
            num_points = count_points_in_las(file_path)

            if num_points > num_points_lim:
                las = laspy.read(file_path)
                points = np.vstack((las.x, las.y, las.z)).T  # Формируем массив точек (N, 3)
                classes = las.classification  # Извлечение классов точек

                # Случайно выбираем 4096 точек
                sampled_points = random_point_sampling(points, num_points_lim)
                
                # Получаем классы для отобранных точек
                sampled_indices = np.random.choice(num_points, num_points_lim, replace=False)
                sampled_classes = classes[sampled_indices]  # Получаем классы для выбранных точек

                # Объединяем координаты и классы
                data_list.append(np.hstack((sampled_points, sampled_classes[:, np.newaxis])))
                points_per_file += 1

                # Записываем в файл, когда накопили достаточно данных
                if points_per_file >= target_points_per_file:
                    chunk_output_path = f"{output_path}_part{file_counter}.npy"
                    np.save(chunk_output_path, np.array(data_list))
                    print(f"Часть {file_counter} сохранена в {chunk_output_path}.")
                    data_list = []  # Очищаем данные для следующей партии
                    points_per_file = 0
                    file_counter += 1

        except Exception as e:
            print(f"Ошибка при обработке файла {las_file}: {e}")

    # Если остались необработанные данные, сохраняем их в последний файл
    if data_list:
        chunk_output_path = f"{output_path}_part{file_counter}.npy"
        np.save(chunk_output_path, np.array(data_list))
        print(f"Оставшаяся часть сохранена в {chunk_output_path}.")
    else:
        print(f"Нет файлов с количеством точек больше {num_points_lim}.")


