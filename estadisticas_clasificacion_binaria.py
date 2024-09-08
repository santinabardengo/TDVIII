import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv('Datos/train_relabeled.csv')
df_train_augmentated = pd.read_csv('Datos/train_augmentated.csv')

def balance_barras(df):
    plt.bar(df['label'].value_counts().index, df['label'].value_counts().values, color=['yellowgreen', 'purple'])
    plt.xlabel('Clase')
    plt.ylabel('Cantidad')
    plt.title('Distribución de muestras')
    plt.xticks([0,1], labels = ['Negativa', 'Positiva'])
    plt.show()

balance_barras(df_train)
balance_barras(df_train_augmentated)