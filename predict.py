import tensorflow as tf
from PIL import Image

from laser_segmentation_quantizer.quantizer import super_fast_prediction, get_quantized_model
from libs_neural import load_imageset, process_images_test
import numpy as np
from tensorflow.keras.preprocessing import image

# Глобальные параметры

IMG_WIDTH = 512               # Ширина картинки
IMG_HEIGHT = 512              # Высота картинки
CLASS_COUNT = 9              # Количество классов на изображении
TEST_DIRECTORY = r'data_img\MUSAC-UNET\musac_only_2012\test'         # Название папки с файлами проверочной выборки
# Назначение цветов классам
class_colors = {
    0: (0, 0, 0),       # Unclassified - Черный
    1: (0, 255, 0),     # Ground - Зеленый
    2: (255, 255, 0),   # Road_markings - Желтый
    3: (0, 0, 255),     # Natural - Синий
    4: (255, 0, 0),     # Building - Красный
    5: (0, 255, 255),   # Utility_line - Бирюзовый
    6: (255, 0, 255),   # Pole - Магента
    7: (255, 128, 0),   # Car - Оранжевый
    8: (128, 128, 128), # Fence - Серый
}

CLASS_LABELS = tuple(class_colors.values())


# 1. Загружаем обученную модель Keras

model = tf.keras.models.load_model('models_saved/model_exp_3_lr_0.0001.23_grey.keras')
# квантуем модель
tflite_model = get_quantized_model(model)

# 2. Загрузка изображения с помощью PIL
test_images = load_imageset(TEST_DIRECTORY, 'original', 'Проверочная')

x_test = []                            # Cписок под проверочную выборку

for img in test_images:                # Для всех изображений выборки:
    x = image.img_to_array(img)       # Перевод изображения в numpy-массив формы: высота x ширина x количество каналов
    x_test.append(x)                   # Добавление элемента в x_train

x_test = np.array(x_test)               # Перевод всей выборки в numpy
print(x_test.shape)


process_images_test(tflite_model, x_test,139)