import pandas as pd

df_train = pd.read_csv('Datos/data_multiclass_model/train_multiclass_relabeled.csv')
df_test = pd.read_csv('Datos/data_multiclass_model/test_multiclass_relabeled.csv')

# Obtenemos datos con etiqueta 0 y 3 de df_train
train_label_0 = df_train[df_train['label'] == 0]
train_label_3 = df_train[df_train['label'] == 3]

# Obtenemos datos con etiqueta 1 y 2 de df_test
test_label_1 = df_test[df_test['label'] == 1]
test_label_2 = df_test[df_test['label'] == 2]

# Balanceamos el test a 4 muestras de cada clase
sample_0 = train_label_0.sample(n=3, random_state=162829)
sample_3 = train_label_3.sample(n=1, random_state=162829)

sample_1 = test_label_1.sample(n=27, random_state=162829)
sample_2 = test_label_2.sample(n=1, random_state=162829)

# Eliminar filas de df_train y de df_test
train_data_prov = df_train.drop(sample_0.index)
train_data_prov = train_data_prov.drop(sample_3.index)
test_data_prov = df_test.drop(sample_1.index)
test_data_prov = test_data_prov.drop(sample_2.index)

# Añadir las filas seleccionadas a test_data
train_data = pd.concat([train_data_prov, sample_1, sample_2])
test_data = pd.concat([test_data_prov, sample_0, sample_3])

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

train_data.to_csv("Datos/data_balanced_multiclass_model_testBalanced/train_multiclass_testBalanced_data.csv", index=False)
test_data.to_csv("Datos/data_balanced_multiclass_model_testBalanced/test_multiclass_testBalanced_data.csv", index=False)