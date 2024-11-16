import pandas as pd

# Clases:
#   0 = Negativo 
#   1 = LSIL + ASC-US
#   2 = HSIL + ASC-H
#   3 = SCC

df_train = pd.read_csv('Datos/original_data/train.csv')
df_classifications = pd.read_csv('Datos/original_data/classifications.csv')
df_test = pd.read_csv('Datos/original_data/test.csv')

def relabel(df, df_classifications, filename):
    df['label'] = 0
    for i, id in enumerate(df['image_id']):

        if sum(df_classifications[df_classifications['image_id'] == id]['bethesda_system'] == 'SCC') > 0:
            df.at[i,'label'] = 3
        elif sum(df_classifications[df_classifications['image_id'] == id]['bethesda_system'] == 'HSIL') > 0 or (sum(df_classifications[df_classifications['image_id'] == id]['bethesda_system'] == 'ASC-H') > 0):
            df.at[i,'label'] = 2
        elif sum(df_classifications[df_classifications['image_id'] == id]['bethesda_system'] == 'LSIL') > 0 or (sum(df_classifications[df_classifications['image_id'] == id]['bethesda_system'] == 'ASC-US') > 0):
            df.at[i,'label'] = 1
    
    df['label'] = df['label'].astype(int)

    df.to_csv(filename, index = False)

relabel(df_train, df_classifications,'Datos/data_multiclass_model/train_multiclass_relabeled.csv')
relabel(df_test, df_classifications,'Datos/data_multiclass_model/test_multiclass_relabeled.csv')

