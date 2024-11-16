from torchvision import transforms
from PIL import Image
import pandas as pd
import torchvision.transforms as T
import os

image_sizes = {
    "EfficientNet-b0": 224,
    "EfficientNet-b1": 240,
    "EfficientNet-b2": 260,
    'EfficientNet-b3': 300,
    'EfficientNet-b4': 380,
}


def resize_image(img, model_name, interpolation):
    size = image_sizes.get(model_name, image_sizes[model_name])
    resize_transform = transforms.Resize((size, size), interpolation=interpolation)
    return resize_transform(img)

def resize_images(data, model_name):

    for row in range(len(data)):
        image_path = 'Datos/images/original_images/' + data['image_filename'][row]
        image = Image.open(image_path)

        # Aplicar cada método de interpolación y guardar la imagen resultante
        for method in interpolation_methods:
            resized_image = resize_image(image, model_name, method)

            # Determinar la carpeta de destino
            if method == transforms.InterpolationMode.NEAREST:
                save_dir = destination_dir_nearest
            elif method == transforms.InterpolationMode.BILINEAR:
                save_dir = destination_dir_bilinear
            elif method == transforms.InterpolationMode.BICUBIC:
                save_dir = destination_dir_bicubic

            # Guardar la imagen en la carpeta correspondiente
            resized_image.save(os.path.join(save_dir, data['image_filename'][row]))

model_name = 'EfficientNet-b0'
interpolation_methods = [T.InterpolationMode.NEAREST, T.InterpolationMode.BILINEAR, T.InterpolationMode.BICUBIC]

# Creamos las rutas de destino para las imágenes interpoladas
destination_dir_nearest = 'Datos/images/' + model_name + '/NearestInterpolation/'
destination_dir_bilinear = 'Datos/images/' + model_name + '/BilinearInterpolation/'
destination_dir_bicubic = 'Datos/images/' + model_name +'/BicubicInterpolation/'

# Creamos las carpetas de destino
os.makedirs(destination_dir_nearest, exist_ok=True)
os.makedirs(destination_dir_bilinear, exist_ok=True)
os.makedirs(destination_dir_bicubic, exist_ok=True)

df_train = pd.read_csv('Datos/original_data/train.csv')
df_test = pd.read_csv('Datos/original_data/test.csv')

resize_images(df_train, model_name)
resize_images(df_test, model_name)