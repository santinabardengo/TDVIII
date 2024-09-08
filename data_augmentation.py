import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

def augment_images(images_paths, save_path, df, filename):
    new_rows = []

    # Obtener el valor máximo de `image_id` para asignar nuevos IDs
    max_image_id = 400

    for idx, image_path in enumerate(images_paths):
        image = Image.open('Datos/images/' + image_path)

        # Definir transformaciones
        rotacion = transforms.RandomRotation(90)
        horizontal_flip = transforms.RandomHorizontalFlip()
        vertical_flip = transforms.RandomVerticalFlip()
        color = transforms.ColorJitter(contrast=0.2, brightness=0.2, saturation=0.2, hue=0.1)

        # Aplicar las transformaciones
        rotated_image = rotacion(image)
        horizontal_image = horizontal_flip(image)
        vertical_image = vertical_flip(image)
        colored_image = color(image)

        rotated_filename = image_path.split('.')[0] + '_rotated.png'
        horizontal_filename = image_path.split('.')[0] + '_horizontal.png'
        vertical_filename = image_path.split('.')[0] + '_vertical.png'
        colored_filename = image_path.split('.')[0] + '_colored.png'

        rotated_image.save(save_path + rotated_filename)
        horizontal_image.save(save_path + horizontal_filename)
        vertical_image.save(save_path + vertical_filename)
        colored_image.save(save_path + colored_filename)

        # Obtener el `label` original y otros metadatos
        original_row = df[df['image_filename'] == image_path].iloc[0]
        
        # Crear nuevas filas con nuevos IDs
        new_rows.append([max_image_id + idx*4 + 1, rotated_filename, original_row['image_doi'], original_row['label']])
        new_rows.append([max_image_id + idx*4 + 2, horizontal_filename, original_row['image_doi'], original_row['label']])
        new_rows.append([max_image_id + idx*4 + 3, vertical_filename, original_row['image_doi'], original_row['label']])
        new_rows.append([max_image_id + idx*4 + 4, colored_filename, original_row['image_doi'], original_row['label']])

    new_df = pd.DataFrame(new_rows, columns=['image_id', 'image_filename', 'image_doi', 'label'])
    
    # Concatenar el DataFrame original con el nuevo DataFrame (sin crear uno separado)
    df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv(filename, index = False)


df_train = pd.read_csv('Datos/train_relabeled.csv')
neg_images_path = df_train[df_train['label'] == 0]['image_filename']

augment_images(neg_images_path, 'Datos/augmented_images/', df_train, 'Datos/train_augmentated.csv')

