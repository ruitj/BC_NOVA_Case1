from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def ball_tree_impute(df, target, n_neighbors=5):
    # Get all features except the target column
    features = df.columns.drop(target)
    
    # Separate rows with and without missing target values
    missing_mask = df[target].isna()
    non_missing_data = df[~missing_mask]
    missing_data = df[missing_mask]

    # Build a ball tree using all features except the target column
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    knn.fit(non_missing_data[features])

    # Find nearest neighbors for rows with missing values
    _, indices = knn.kneighbors(missing_data[features])

     # Initialize a Series to store imputed values
    imputed_values = pd.Series(index=df.index)

    # Impute missing values by averaging the target values of nearest neighbors
    for i, neighbor_indices in enumerate(indices):
        # Calculate the mean of the neighbors' target values
        mean_value = non_missing_data.iloc[neighbor_indices][target].mean()
        imputed_values[missing_data.index[i]] = mean_value

    # Combine the imputed values with the original target values
    result = df[target].combine_first(imputed_values)

    return result


## Outliers


def detect_outliers_iqr(df, missing_threshold):
    missing_col = []
    outliers_indices = set()
    bounds = {}  
    
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Store the bounds
        bounds[column] = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
        
        # Identify outliers
        outlier_data = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers_indices.update(outlier_data.index)
        
        missing = len(outlier_data) / len(df) * 100
        
        # Print the number of outliers
        print(f'Column: {column} - Number of Outliers: {len(outlier_data)}')
        print(f'Column: {column} - % of Outliers: {missing:.2f}% \n')
        
        if missing > missing_threshold:
            missing_col.append(column)
        
        # Boxplot for each column
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x=column, color='orange', showfliers=False)  
        sns.stripplot(
            data=outlier_data, 
            x=column, 
            color='red', 
            jitter=True, 
            label='Outliers'
        )
        plt.title(f'Boxplot with Outliers for {column}')
        plt.legend()
        plt.show()
    
    print(f'Columns with more than {missing_threshold}% Outliers:')        
    print(missing_col)
    
    return bounds  
