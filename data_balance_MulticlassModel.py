import pandas as pd

df_train = pd.read_csv('Datos/data_multiclass_model/train_multiclass_relabeled.csv')
df_test = pd.read_csv('Datos/data_multiclass_model/test_multiclass_relabeled.csv')

train_label_0 = df_train[df_train['label'] == 0]

# Seleccionar 2 filas aleatorias
samples = train_label_0.sample(n=2, random_state=162829)

# Eliminar esas filas del df_train
train_data = df_train.drop(samples.index)

# Añadir las filas seleccionadas al df_test
test_data = pd.concat([df_test, samples])

test_data.reset_index(drop=True, inplace=True)

train_data.to_csv("Datos/data_balanced_multiclass_model/train_multiclass_data.csv", index=False)
test_data.to_csv("Datos/data_balanced_multiclass_model/test_multiclass_data.csv", index=False)