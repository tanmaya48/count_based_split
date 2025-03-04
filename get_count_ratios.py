import pandas as pd

def get_class_proportions(df):
    return df.sum()/len(df)

val_df = pd.read_csv('coco_counts_val.csv').select_dtypes(include=["number"])
train_df = pd.read_csv('coco_counts_train.csv').select_dtypes(include=["number"])


val_prop = get_class_proportions(val_df)
train_prop = get_class_proportions(train_df)


for col in val_df.columns:
    print(f"{col} {val_prop[col]/train_prop[col]} ")
