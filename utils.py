import os
import pandas as pd
from tqdm import tqdm

def get_counts_table(label_dir,num_classes=80):
    object_counts_dict = {}
    for filename in tqdm(os.listdir(label_dir)):
        counts = [0] * num_classes
        with open(os.path.join(label_dir,filename),'r') as f:
            data = f.readlines()
            for line in data:
                cls = int(line.split(' ')[0])
                counts[cls]+=1
        object_counts_dict[filename] = counts
    df = pd.DataFrame.from_dict(object_counts_dict, orient='index', columns =[f"class_{i}" for i in range(num_classes)])
    return df



def create_standardized_columns(table,numerical_columns):
    df = table.copy()
    for col in numerical_columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df




if __name__ == '__main__':
    label_dir = 'coco2017/coco_converted/labels/val2017/'
    label_table = get_counts_table(label_dir)
    label_table.to_csv('coco_counts_val.csv')

