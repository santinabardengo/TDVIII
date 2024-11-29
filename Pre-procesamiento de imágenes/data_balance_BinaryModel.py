import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

df_train = pd.read_csv('../Datos/data_binary_model/train_binary_relabeled.csv')
df_test = pd.read_csv('../Datos/data_binary_model/test_binary_relabeled.csv')


plt.figure(figsize=(6, 4))
sns.countplot(data=df_test, x='label', palette='plasma', width=0.5)
plt.title('Distribución de labels en el conjunto de prueba')
plt.xlabel('Clase')
plt.ylabel('Cantidad')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df_train, x='label', palette='viridis', width=0.5)
plt.title('Distribución de labels en el conjunto de entrenamiento')
plt.xlabel('Clase')
plt.ylabel('Cantidad')
plt.show()

train_negatives = df_train[df_train['label'] == 0]

# Seleccionar 10 filas aleatorias
negative_samples = train_negatives.sample(n=10, random_state=162829)

# Eliminar esas filas del train_data original
train_data = df_train.drop(negative_samples.index)

# Añadir las filas seleccionadas al test_data
test_data = pd.concat([df_test, negative_samples])

test_data.reset_index(drop=True, inplace=True)

train_data.to_csv("Datos/data_balanced_binary_model/train_binary_data.csv", index=False)
test_data.to_csv("Datos/data_balanced_binary_model/test_binary_data.csv", index=False)