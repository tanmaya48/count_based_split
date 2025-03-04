import numpy as np
import pandas as pd
from utils import create_standardized_columns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

seed = 10
subset_proportion = 0.02

def main(table_path):
    df = pd.read_csv(table_path)
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    df_st = create_standardized_columns(df,numerical_columns)
    #
    proportions = (df[numerical_columns].sum()/len(df)).to_list()
    inertia_scores = []
    k_values = range(2,30) 
    kmeans_models = []
    #
    for k in tqdm(k_values):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        kmeans.fit(df_st[numerical_columns])
        kmeans_models.append(kmeans)
        inertia_scores.append(kmeans.inertia_) 
    inertia_scores = np.array(inertia_scores) 
    diffs = inertia_scores[:-1] - inertia_scores[1:]
    best_model = kmeans_models[int(np.argmin(diffs))]
    df['cluster_id'] = best_model.predict(df_st[numerical_columns])
    remaining_df, subset_df = train_test_split(counts_df,
                                               test_size=subset_proportion, 
                                               random_state=seed, 
                                               stratify=df['cluster_id'])
    subset_df.to_csv('stratified_subset.csv')
    random_remaining_df, random_subset_df = train_test_split(counts_df,test_size=subset_proportion,random_state=seed)
    random_subset_df.to_csv('random_stratified_subset.csv')


if __name__ == '__main__':
    table_path = 'coco_counts.csv'
    main(table_path)
