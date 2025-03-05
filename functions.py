import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec

from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import RegularPolygon, Ellipse
from matplotlib import cm, colorbar
from matplotlib import colors as mpl_colors

# MISSING DATA FUNCTION
def missing_data(df):
    """
    Gives the count and percentage of missing values for each column in a DataFrame
    """    
    # Number of missing values in each column
    missing_count = df.isnull().sum()
    
    # Percentage of missing values for each column
    missing_percentage = ((missing_count / df.shape[0]) * 100).round(2)
    
    missing_data = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing %': missing_percentage
    })
    
    # Show only columns with missing values
    missing_data = missing_data[missing_data['Missing Count'] > 0]
    
    # Sort in descending order
    missing_data = missing_data.sort_values(by='Missing Count', ascending=False)
    return missing_data


# IQR OUTLIER FUNCTION
def remove_outliers_iqr(df, columns, threshold=1.5):
    rows_removed = {}  
    total_removed=0
    initial_rows = df.shape[0]
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        removed = df[~((df[column] >= lower_bound) & (df[column] <= upper_bound))].shape[0]
        rows_removed[column] = removed
        total_removed += removed
        
        print(f'Upper Bound for {column}: {upper_bound}')
        print(f'Lower Bound for {column}: {lower_bound}')
        print(50*'-')

        
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f'Rows removed for each:{rows_removed}')
    print(f'Total removed:{total_removed}')
    print(f'Percentage removed:{round((total_removed/initial_rows)*100,4)}%')
    return df


# PERCENTILE OUTLIER FUNCTION
def remove_outliers_percentile(df, columns, lower_percentile=0, upper_percentile=95):
    rows_removed = {}  # Dictionary to store the number of rows removed per column
    total_removed = 0
    initial_rows = df.shape[0]
    
    for column in columns:
        lower_bound = df[column].quantile(lower_percentile / 100)
        upper_bound = df[column].quantile(upper_percentile / 100)
        
        removed = df[~((df[column] >= lower_bound) & (df[column] <= upper_bound))].shape[0]
        rows_removed[column] = removed
        total_removed += removed
        
        print(f'Upper Bound for {column}: {upper_bound}')
        print(f'Lower Bound for {column}: {lower_bound}')
        print(50 * '-')
        
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    print(f'Rows removed for each: {rows_removed}')
    print(f'Total removed: {total_removed}')
    print(f'Percentage removed: {round((total_removed / initial_rows) * 100, 4)}%')
    
    return df



# Z-SCORE OUTLIER
import numpy as np
from scipy.stats import zscore

def remove_outliers_zscore(df, column, threshold=3):
    rows_removed = {}  
    total_removed = 0
    initial_rows = df.shape[0]

    # Calculate Z-scores for the column
    z_scores = zscore(df[column])

    # Identify outliers based on the Z-score threshold
    outliers = np.abs(z_scores) > threshold

    # Count the removed outliers
    removed = df[outliers].shape[0]
    rows_removed[column] = removed
    total_removed += removed

    print(f"Total outliers in column {column}: {removed}")

    # Clean the dataframe by removing outliers
    df_cleaned = df[~outliers]

    # Calculate upper and lower bounds based on Z-score
    mean_value = df[column].mean()
    std_dev = df[column].std()

    lower_bound = mean_value - threshold * std_dev
    upper_bound = mean_value + threshold * std_dev

    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")
    print(f"Percentage of data removed: {round((total_removed / initial_rows) * 100, 4)}%")
    
    return df_cleaned



# -----------------------------------------------------------------------------------------------
# Functions for plotting


def plot_distribution_and_boxplot(df, columns_with_outliers, w=15,h=3):
    """
    Plots the distribution and boxplot for a list of columns with outliers.
    """
    plt.figure(figsize=(w, h*len(columns_with_outliers)))

    for i, column in enumerate(columns_with_outliers):
        # Distribution plot
        plt.subplot(len(columns_with_outliers), 2, 2 * i + 1)
        sns.histplot(df[column], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

        # Boxplot
        plt.subplot(len(columns_with_outliers), 2, 2 * i + 2)
        sns.boxplot(x=df[column], color='orange')
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)

    plt.tight_layout()
    plt.show()



def plot_distribution_grid(df, subset_num, cols=2, title="Feature Distributions"):
    sns.set(style="white")
    rows = math.ceil(len(subset_num) / cols)
    fig = plt.figure(figsize=(cols * 9, rows * 5))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.94) 
    outer = gridspec.GridSpec(rows, cols, wspace=0.3, hspace=0.5)

    for i, feature in enumerate(subset_num):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            if j == 0:
                img = df.copy()
                img.dropna(subset=[feature], inplace=True)
                bin_edges = np.histogram_bin_edges(img[feature], bins='auto')
                if img[feature].dtype == int:
                    bin_edges = np.arange(int(bin_edges.min()), int(bin_edges.max()) + 1)
                sns.histplot(img[feature], bins=bin_edges, kde=False, ax=ax, color="lightblue", edgecolor="gray", alpha=0.7)
                
                if img[feature].dtype == int:
                    ax.axvline(img[feature].mode()[0], color='orange', linestyle='--', label=f"mode: {round(img[feature].mode()[0])}", alpha=0.8)    
                ax.axvline(img[feature].median(), color='gray', linestyle='--', label=f'median: {round(img[feature].median())}', alpha=0.8)
                
                
                ax.legend()
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_title(f'Distribution of {feature}')
                ax.grid(True, linestyle='-', alpha=0.6, axis='both')
            else:
                sns.boxplot(x=img[feature], ax=ax, color="lightblue", width=0.25,
                            boxprops=dict(alpha=0.5), flierprops=dict(marker='o', alpha=0.35))
                ax.grid(True, linestyle='-', alpha=0.6)
            fig.add_subplot(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.show()






def compare_figure_outliers(df_original, df, num_feats):
    sns.set_style('whitegrid')
    frows = math.ceil(len(num_feats) / 2)
    fcols = 2
    
    fig = plt.figure(figsize=(15, 5 * frows))
    
    subfigs = fig.subfigures(frows, fcols, wspace=0.03, hspace=0.03)
    
    for sfig, feat in zip(subfigs.flatten(), num_feats):
        axes = sfig.subplots(2, 1, sharex=True)
        
        sns.boxplot(x=df_original[feat], ax=axes[0])
        axes[0].set_ylabel("Original")
        axes[0].set_title(feat, fontsize="large")
        
        sns.boxplot(x=df[feat], ax=axes[1])
        axes[1].set_ylabel("Outliers\nRemoved")
        axes[1].set_xlabel("")
        
        sfig.set_facecolor("#F9F9F9")
        sfig.subplots_adjust(left=0.2, right=0.95, bottom=0.1)
        
    plt.show()
    sns.set()



def plot_numerical(df, col):
    sns.set(style="white")
    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1.7], hspace=0.03)
    
    col_median = np.median(df[col])
    
    ax1 = fig.add_subplot(gs[0, 0])
    bin_edges = np.histogram_bin_edges(df[col], bins='auto')
    if df[col].dtype == int:
        bin_edges = np.arange(int(bin_edges.min()), int(bin_edges.max()) + 1)

    ax1.hist(df[col], bins=bin_edges, alpha=0.9, color="lightblue", edgecolor="gray")
    
    if df[col].dtype == int:
        ax1.axvline(df[col].mode()[0], color='orange', linestyle='--', label=f"mode: {round(df[col].mode()[0])}", alpha=0.8)    
    ax1.axvline(col_median, color='lightgray', linestyle='--', label=f"median: {round(col_median)}", alpha=0.8)
    
    ax1.set_xticks([])
    ax1.set_ylabel("frequency")
    ax1.legend()
    ax1.grid(True, linestyle='-', alpha=0.6)
    ax2 = fig.add_subplot(gs[1, 0])
    sns.boxplot(x=df[col], ax=ax2, color="lightblue", width=0.25,
                boxprops=dict(alpha=0.5), flierprops=dict(marker='o', alpha=0.35))
    ax2.set_xlabel(col)
    ax2.set_yticks([]) 
    ax2.grid(True, linestyle='-', alpha=0.6)

    plt.suptitle(f"'{col}' distribution", fontsize=14, fontweight='bold')
    plt.show()
    


# Criar para scaling: ALTERAR
# def compare_figure_scaling(df_original, df_scaled, num_feats):
#     sns.set_style('whitegrid')
#     frows = math.ceil(len(num_feats) / 2)
#     fcols = 2
    
#     fig = plt.figure(figsize=(15, 5 * frows))
    
#     subfigs = fig.subfigures(frows, fcols, wspace=0.03, hspace=0.03)
    
#     for sfig, feat in zip(subfigs.flatten(), num_feats):
#         axes = sfig.subplots(2, 1, sharex=True)
        
#         # Original data boxplot
#         sns.boxplot(x=df_original[feat], ax=axes[0])
#         axes[0].set_ylabel("Original")
#         axes[0].set_title(feat, fontsize="large")
        
#         # Scaled data boxplot
#         sns.boxplot(x=df_scaled[feat], ax=axes[1])
#         axes[1].set_ylabel("Scaled")
#         axes[1].set_xlabel("")
        
#         sfig.set_facecolor("#F9F9F9")
#         sfig.subplots_adjust(left=0.2, right=0.95, bottom=0.1)
        
#     plt.show()
#     sns.set()



def compare_figure_scaling_histograms(df_original, df_scaled, num_feats):
    sns.set_style('whitegrid')
    frows = math.ceil(len(num_feats) / 2)
    fcols = 2
    
    # Create figure
    fig = plt.figure(figsize=(15, 5 * frows))
    
    subfigs = fig.subfigures(frows, fcols, wspace=0.03, hspace=0.03)
    
    # Loop through features and plot
    for sfig, feat in zip(subfigs.flatten(), num_feats):
        axes = sfig.subplots(2, 1, sharex=True)
        
        # Plot the histogram for the original data
        sns.histplot(df_original[feat], bins=15, kde=True, ax=axes[0])
        axes[0].set_title(f"{feat} - Original", fontsize="large")
        axes[0].set_ylabel("Frequency")
        
        # Plot the histogram for the scaled data
        sns.histplot(df_scaled[feat], bins=15, kde=True, ax=axes[1])
        axes[1].set_title(f"{feat} - Scaled", fontsize="large")
        axes[1].set_ylabel("Frequency")
        
        sfig.set_facecolor("#F9F9F9")
        sfig.subplots_adjust(left=0.2, right=0.95, bottom=0.1)
        
    plt.show()
    sns.set()




def plot_categorical_distributions(df, cat_cols, cols_num=2):
    if len(cat_cols) == 1:
        order = df[cat_cols[0]].value_counts().index
        plt.figure(figsize=(8, 6))
        sns.countplot(x=cat_cols[0], data=df,order=order)
        plt.title(f'Distribution of {cat_cols[0]}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        rows = math.ceil(len(cat_cols) / cols_num)

        fig, axes = plt.subplots(rows, cols_num, figsize=(cols_num * 7, rows *5))

        axes = axes.flatten() if len(cat_cols) > 1 else [axes]

        for ax, feat in zip(axes, cat_cols):
            order = df[feat].value_counts().index
            sns.countplot(x=feat, data=df, ax=ax, order=order)
            ax.set_title(f'Distribution of {feat}')
            ax.tick_params(axis='x', rotation=45)

        for i in range(len(cat_cols), len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle("Categorical Variables Distribution", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



def iqr_or_percentile(df, columns, iqr_multiplier=3, lower_quantile=0.01, upper_quantile=0.99):
    cols_remove_right_outliers_perc = []
    cols_remove_right_outliers_IQR = []
    cols_remove_left_outliers_IQR = []
    cols_remove_left_outliers_perc = []
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound_IQR = Q1 - iqr_multiplier * IQR
        upper_bound_IQR = Q3 + iqr_multiplier * IQR
        lower_bound_perc = df[col].quantile(lower_quantile)
        upper_bound_perc = df[col].quantile(upper_quantile)
        
        # Identify right outliers using percentiles and IQR
        if df[col].max() > upper_bound_perc and upper_bound_perc > upper_bound_IQR:
            cols_remove_right_outliers_perc.append(col)
        elif df[col].max() > upper_bound_IQR and upper_bound_IQR > upper_bound_perc:
            cols_remove_right_outliers_IQR.append(col)
        
        # Identify left outliers using IQR
        if df[col].min() < lower_bound_IQR:
            cols_remove_left_outliers_IQR.append(col)
        
        # Identify left outliers using percentiles
        elif df[col].min() < lower_bound_perc and lower_bound_perc < lower_bound_IQR:
            cols_remove_left_outliers_perc.append(col)
    
    # Print the results for verification
    print("Columns with right outliers (percentile):", cols_remove_right_outliers_perc)
    print("Columns with right outliers (IQR):", cols_remove_right_outliers_IQR)
    print("Columns with left outliers (IQR):", cols_remove_left_outliers_IQR)
    print("Columns with left outliers (percentile):", cols_remove_left_outliers_perc)

    return (
        cols_remove_right_outliers_perc,
        cols_remove_right_outliers_IQR,
        cols_remove_left_outliers_IQR,
        cols_remove_left_outliers_perc,
    )




def k_distance_graph(df, metric_features):
    min_pts = 2 * len(metric_features)
    neigh = NearestNeighbors(n_neighbors=min_pts)
    neigh.fit(df[metric_features])

    distances, _ = neigh.kneighbors(df[metric_features])

    distances = np.sort(distances[:, -1])

    plt.figure(figsize=(8, 6))
    plt.plot(distances, color='orange', linestyle='-', linewidth=2)
    plt.title("K-Distance graph to find the right epsilon", fontsize=16, fontweight='bold')
    plt.xlabel("Sorted data points", fontsize=14)
    plt.ylabel(f"Distance to {min_pts-1}th nearest neighbor", fontsize=14)
    plt.tight_layout()  
    plt.show()



# Clustering functions


def get_ss(df, feats):
    """
    Calculate the sum of squares (SS) for the given DataFrame.

    The sum of squares is computed as the sum of the variances of each column
    multiplied by the number of non-NA/null observations minus one.

    Parameters:
    df (pandas.DataFrame): The input DataFrame for which the sum of squares is to be calculated.
    feats (list of str): A list of feature column names to be used in the calculation.

    Returns:
    float: The sum of squares of the DataFrame.
    """
    df_ = df[feats]
    ss = np.sum(df_.var() * (df_.count() - 1))
    
    return ss 


def get_ssb(df, feats, label_col):
    """
    Calculate the between-group sum of squares (SSB) for the given DataFrame.
    The between-group sum of squares is computed as the sum of the squared differences
    between the mean of each group and the overall mean, weighted by the number of observations
    in each group.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column in the DataFrame that contains the group labels.
    
    Returns
    float: The between-group sum of squares of the DataFrame.
    """
    
    ssb_i = 0
    for i in np.unique(df[label_col]):
        df_ = df.loc[:, feats]
        X_ = df_.values
        X_k = df_.loc[df[label_col] == i].values
        
        ssb_i += (X_k.shape[0] * (np.square(X_k.mean(axis=0) - X_.mean(axis=0))) )

    ssb = np.sum(ssb_i)
    

    return ssb


def get_ssw(df, feats, label_col):
    """
    Calculate the sum of squared within-cluster distances (SSW) for a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing cluster labels.

    Returns:
    float: The sum of squared within-cluster distances (SSW).
    """
    feats_label = feats+[label_col]

    df_k = df[feats_label].groupby(by=label_col).apply(lambda col: get_ss(col, feats), 
                                                       include_groups=False)

    return df_k.sum()


def get_rsq(df, feats, label_col):
    """
    Calculate the R-squared value for a given DataFrame and features.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    feats (list): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing the labels or cluster assignments.

    Returns:
    float: The R-squared value, representing the proportion of variance explained by the clustering.
    """

    df_sst_ = get_ss(df, feats)                 # get total sum of squares
    df_ssw_ = get_ssw(df, feats, label_col)     # get ss within
    df_ssb_ = df_sst_ - df_ssw_                 # get ss between

    # r2 = ssb/sst 
    return (df_ssb_/df_sst_)
    
def get_r2_hc(df, link_method, max_nclus, min_nclus=1, dist="euclidean"):
    """This function computes the R2 for a set of cluster solutions given by the application of a hierarchical method.
    The R2 is a measure of the homogenity of a cluster solution. It is based on SSt = SSw + SSb and R2 = SSb/SSt. 
    
    Parameters:
    df (DataFrame): Dataset to apply clustering
    link_method (str): either "ward", "complete", "average", "single"
    max_nclus (int): maximum number of clusters to compare the methods
    min_nclus (int): minimum number of clusters to compare the methods. Defaults to 1.
    dist (str): distance to use to compute the clustering solution. Must be a valid distance. Defaults to "euclidean".
    
    Returns:
    ndarray: R2 values for the range of cluster solutions
    """
    
    r2 = []  # where we will store the R2 metrics for each cluster solution
    feats = df.columns.tolist()
    
    for i in range(min_nclus, max_nclus+1):  # iterate over desired ncluster range
        cluster = AgglomerativeClustering(n_clusters=i, metric=dist, linkage=link_method)
        
        #get cluster labels
        hclabels = cluster.fit_predict(df) 
        
        # concat df with labels
        df_concat = pd.concat([df, pd.Series(hclabels, name='labels', index=df.index)], axis=1)  
        
        
        # append the R2 of the given cluster solution
        r2.append(get_rsq(df_concat, feats, 'labels'))
        
    return np.array(r2)


def get_silhouette_hc(df, link_method, max_nclus, min_nclus=2, dist="euclidean"):
    """
    Computes the Silhouette Score for a set of cluster solutions using hierarchical clustering.
    
    Parameters:
    df (DataFrame): Dataset to apply clustering
    link_method (str): either "ward", "complete", "average", "single"
    max_nclus (int): Maximum number of clusters to compare the methods
    min_nclus (int): Minimum number of clusters to compare the methods. Defaults to 2.
    dist (str): Distance metric to use for clustering. Must be a valid metric supported by sklearn. Defaults to "euclidean".

    Returns:
    ndarray: Silhouette scores for the range of cluster solutions
    """

    silhouette_scores = []  # Store the Silhouette Scores for each cluster solution

    for i in range(min_nclus, max_nclus + 1):  # Iterate over the desired cluster range
        # Perform Agglomerative Clustering
        cluster = AgglomerativeClustering(n_clusters=i, metric=dist, linkage=link_method)
        
        # Get cluster labels
        hclabels = cluster.fit_predict(df)
        
        # Compute Silhouette Score for the given cluster solution
        score = silhouette_score(df, hclabels, metric=dist)
        silhouette_scores.append(score)

    return np.array(silhouette_scores)


def plot_hexagons(som,              # Trained SOM model 
                  sf,               # matplotlib figure object
                  colornorm,        # colornorm
                  matrix_vals,      # SOM weights or
                  label="",         # title for figure
                  cmap=cm.Grays,    # colormap to use
                  annot=False       
                  ):

    
    axs = sf.subplots(1,1)
    
    for i in range(matrix_vals.shape[0]):
        for j in range(matrix_vals.shape[1]):

            wx, wy = som.convert_map_to_euclidean((i,j)) 

            hex = RegularPolygon((wx, wy), 
                                numVertices=6, 
                                radius= np.sqrt(1/3),
                                facecolor=cmap(colornorm(matrix_vals[i, j])), 
                                alpha=1, 
                                edgecolor='white',
                                linewidth=.5)
            axs.add_patch(hex)
            if annot==True:
                annot_val = np.round(matrix_vals[i,j],2)
                if int(annot_val) == annot_val:
                    annot_val = int(annot_val)
                axs.text(wx,wy, annot_val, 
                        ha='center', va='center', 
                        fontsize='x-small')


    ## Remove axes for hex plot
    axs.margins(.05)
    axs.set_aspect('equal')
    axs.axis("off")
    axs.set_title(label)

    

    # ## Add colorbar
    divider = make_axes_locatable(axs)
    ax_cb = divider.append_axes("right", size="5%", pad="0%")

    ## Create a Mappable object
    cmap_sm = plt.cm.ScalarMappable(cmap=cmap, norm=colornorm)
    cmap_sm.set_array([])

    ## Create custom colorbar 
    cb1 = colorbar.Colorbar(ax_cb,
                            orientation='vertical', 
                            alpha=1,
                            mappable=cmap_sm
                            )
    cb1.ax.get_yaxis().labelpad = 6

    # Add colorbar to plot
    sf.add_axes(ax_cb)




    return sf 
