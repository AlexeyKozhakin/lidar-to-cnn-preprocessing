import os
import laspy
import matplotlib.pyplot as plt

def count_points_in_las(file_path):
    """Возвращает количество точек в LAS файле."""
    las = laspy.read(file_path)
    return len(las.points)

def process_las_files_in_directory(directory):
    """Считывает все LAS файлы в каталоге и возвращает список с количеством точек в каждом."""
    points_count = []
    las_files = [f for f in os.listdir(directory) if f.endswith('.las')]
    
    for las_file in las_files:
        file_path = os.path.join(directory, las_file)
        num_points = count_points_in_las(file_path)
        points_count.append(num_points)
        print(f"Файл {las_file}: {num_points} точек")
    
    return points_count, las_files  # Возвращаем также имена файлов

def remove_files_below_threshold(directory, files, threshold):
    """Удаляет файлы, содержащие меньше заданного порога точек."""
    for las_file in files:
        file_path = os.path.join(directory, las_file)
        num_points = count_points_in_las(file_path)
        
        if num_points < threshold:
            os.remove(file_path)
            print(f"Удалён файл {las_file}: {num_points} точек меньше порога {threshold}")

def plot_histogram(data):
    """Строит гистограмму по количеству точек в LAS файлах."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=20, edgecolor='black')
    plt.title('Гистограмма количества точек в LAS файлах')
    plt.xlabel('Количество точек')
    plt.ylabel('Количество файлов')
    plt.grid(True)
    plt.show()

# Укажите путь к каталогу с LAS файлами
directory_path = r'D:\data\las_org\data_las_stpls3d\all_org_las_cut_32'
threshold = 4096  # Задайте пороговое значение

# Считываем файлы и получаем количество точек
points_counts, las_files = process_las_files_in_directory(directory_path)

# Удаляем файлы с количеством точек ниже порога
#remove_files_below_threshold(directory_path, las_files, threshold)

# Строим гистограмму
plot_histogram(points_counts)
