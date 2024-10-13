# Служебная функция загрузки выборки изображений из файлов в папке
from laser_segmentation_quantizer.quantizer import quantized_model_predicton
import time
import numpy as np
import os
from PIL import Image
# Инструменты для работы с изображениями
from tensorflow.keras.preprocessing import image

def load_imageset(folder,   # имя папки
                  subset,   # подмножество изображений - оригинальные или сегментированные
                  title,     # имя выборки
                  IMG_WIDTH=512, IMG_HEIGHT=512):

    # Cписок для хранения изображений выборки
    image_list = []

    # Отметка текущего времени
    cur_time = time.time()

    # Для всех файлов в каталоге по указанному пути:
    for filename in sorted(os.listdir(f'{folder}/{subset}')):

        # Чтение очередной картинки и добавление ее в список изображений требуемого размера
        image_list.append(image.load_img(os.path.join(f'{folder}/{subset}', filename),
                                         target_size=(IMG_WIDTH, IMG_HEIGHT)))

    # Вывод времени загрузки картинок выборки
    print('{} выборка загружена. Время загрузки: {:.2f} с'.format(title,
                                                                  time.time() - cur_time))

    # Вывод количества элементов в выборке
    print('Количество изображений:', len(image_list))

    return image_list

# Функция преобразования тензора меток класса в цветное сегметрированное изображение

def labels_to_rgb(image_list,  # список одноканальных изображений
                 IMG_WIDTH=512, IMG_HEIGHT=512):
    class_colors = {
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

    CLASS_LABELS = list(class_colors.values())

    result = []

    # Для всех картинок в списке:
    for y in image_list:
        # Создание пустой цветной картики
        temp = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), dtype='uint8')

        # По всем классам:
        for i, cl in enumerate(CLASS_LABELS):
            # Нахождение пикселов класса и заполнение цветом из CLASS_LABELS[i]
            temp[np.where(np.all(y==i, axis=-1))] = CLASS_LABELS[i]

        result.append(temp)

    return np.array(result)

# Функция визуализации процесса сегментации изображений
def process_images_test(tflite_model,
                        x_test, # обученная модель
                        count = 1,    # количество случайных картинок для сегментации
                        save_dir='predictions'  # директория для сохранения изображений
                       ):

    # Создание директории, если её не существует
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Генерация случайного списка индексов в количестве count между (0, len(x_val)
    indexes = np.arange(count)

    # Вычисление предсказания сети для картинок с отобранными индексами
    predict = []
    for ind in indexes:
        print('Тип входа:', type(x_test[ind:ind+1]))
        print('Размер входа:', x_test[ind:ind+1].shape)
        # Начало измерения времени
        start_time = time.time()
        predict.append(np.argmax(quantized_model_predicton(tflite_model, x_test[ind:ind+1]), axis=-1))
        end_time = time.time()
        # Расчет времени выполнения
        execution_time = end_time - start_time
        # Вывод времени предсказания
        print(f"Время выполнения предсказания: {execution_time:.6f} секунд")

    predict = np.array(predict)
    # Удаление лишнего измерения
    predict = np.squeeze(predict, axis=1)  # Теперь форма будет (2, 512, 512)
    print('Тип predict:', type(predict))
    print('Размер predict:', predict.shape)
    # Подготовка цветов классов для отрисовки предсказания
    orig = labels_to_rgb(predict[..., None])
    #orig = labels_to_rgb(predict)

    # Отрисовка результата работы модели
    for i in range(count):
        pred_image = Image.fromarray(orig[i])
        pred_image.save(os.path.join(save_dir, f'predicted_image_{indexes[i]}.png'))


