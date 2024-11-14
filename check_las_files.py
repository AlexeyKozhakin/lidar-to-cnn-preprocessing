import laspy

work_with_las = True
work_with_ply = False


if work_with_las:
    # Загрузка файла LAS
    las_file_path = r"D:\data\las_org\data_las_stpls3d\all_org_las_rgb\RA_points.las"  # Замените на путь к вашему файлу
    las = laspy.read(las_file_path)

    # Выводим основные атрибуты и их значения
    print("Основные атрибуты LAS файла:")
    print("X:", max(las.red))
    print("Y:", las.green)
    print("Z:", las.blue)

    # Получаем список всех доступных атрибутов, кроме X, Y, Z
    additional_attributes = [dimension for dimension in las.point_format.dimension_names if dimension not in ('x', 'y', 'z')]

    print("\nДругие доступные атрибуты в LAS файле:")
    for attribute in additional_attributes:
        print(f"{attribute}: {getattr(las, attribute)}")


if work_with_ply:
    import open3d as o3d
    import numpy as np

    # Загрузка файла .ply
    ply_file_path = r"C:\Users\alexe\Downloads\UM\work\Data\STPLS3D_ply\STPLS3D_ply\STPLS3D\RealWorldData\RA_points.ply"  # Замените на путь к вашему файлу
    ply_data = o3d.io.read_point_cloud(ply_file_path)

    # Извлечение координат точек
    points = np.asarray(ply_data.points)

    # Проверка наличия цвета
    if ply_data.has_colors():
        colors = np.asarray(ply_data.colors)
        print("Цветовые атрибуты (R, G, B):")
        print("Red (max):", colors[:, 0].max())
        print("Green (max):", colors[:, 1].max())
        print("Blue (max):", colors[:, 2].max())
    else:
        print("Цветовые атрибуты отсутствуют в файле .ply")

    # Проверка наличия нормалей
    if ply_data.has_normals():
        normals = np.asarray(ply_data.normals)
        print("\nНормали доступны для точек.")

    # Выводим основные атрибуты
    print("\nОсновные атрибуты PLY файла:")
    print("X:", points[:, 0].max())
    print("Y:", points[:, 1].max())
    print("Z:", points[:, 2].max())

    # Если нужны дополнительные атрибуты (возможно с использованием pandas)
    # конвертируем точки и цвета в DataFrame
    try:
        import pandas as pd

        df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
        if ply_data.has_colors():
            color_df = pd.DataFrame(colors, columns=['Red', 'Green', 'Blue'])
            df = pd.concat([df, color_df], axis=1)
        print("\nПервые несколько строк данных PLY файла:\n", df.head())
    except ImportError:
        print("Pandas не установлен. Установите его для работы с таблицами данных.")
