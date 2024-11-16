import pandas as pd

# Clases:
#   0: Negativo
#   1: Positivo

df_train = pd.read_csv('Datos/original_data/train.csv')
df_classifications = pd.read_csv('Datos/original_data/classifications.csv')
df_test = pd.read_csv('Datos/original_data/test.csv')

def relabel(df, df_classifications, filename):

    df['label'] = 0

    for i, id in enumerate(df['image_id']):
        if sum(df_classifications[df_classifications['image_id'] == id]['bethesda_system'] != 'Negative for intraepithelial lesion') > 0:
            df.at[i,'label'] = 1
    
    df['label'] = df['label'].astype(int)

    df.to_csv(filename, index = False)

relabel(df_train, df_classifications,'Datos/data_binary_model/train_relabeled.csv')
relabel(df_test, df_classifications,'Datos/data_binary_model/test_relabeled.csv')

