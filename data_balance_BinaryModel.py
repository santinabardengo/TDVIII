import pandas as pd

df_train = pd.read_csv('Datos/data_binary_model/train_binary_relabeled.csv')
df_test = pd.read_csv('Datos/data_binary_model/test_binary_relabeled.csv')

train_negatives = df_train[df_train['label'] == 0]

# Seleccionar 10 filas aleatorias
negative_samples = train_negatives.sample(n=10, random_state=162829)

# Eliminar esas filas del train_data original
train_data = df_train.drop(negative_samples.index)

# Añadir las filas seleccionadas al test_data
test_data = pd.concat([df_test, negative_samples])

test_data.reset_index(drop=True, inplace=True)

train_data.to_csv("Datos/data_balanced_binary_model/train_data.csv", index=False)
test_data.to_csv("Datos/data_balanced_binary_model/test_data.csv", index=False)