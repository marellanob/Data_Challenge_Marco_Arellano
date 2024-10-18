import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.stats as stats

#########################---------------------------------------------------------------------

# This Utils page contains functions for the Clustering process.
# Ensure this file is in the same directory as the project. 
# If not, Jupyter Notebook won't function properly. Thank you!

#-----------------------------------------------------------------------------------------------

#####################-----------------------------------
## IDENTIFY SKEWED VARIABLES
#####################-----------------------------------

# Objective: Identify skewed variables to perform the right scaling process before the PCA
def identify_right_skewed(df, skew_threshold=0.5):
    
    # Calculate skewness for numeric columns
    skewness = df.select_dtypes(include='number').skew()
    
    # Identify right-skewed variables (skewness > skew_threshold)
    right_skewed_cols = skewness[skewness > skew_threshold].index.tolist()
    
    # Identify no-skewed variables 
    not_skewed_columns = [col for col in df.columns if col not in right_skewed_cols]
    
    return right_skewed_cols, not_skewed_columns


#-----------------------------------------------------------------------------------------------

#####################-----------------------------------
## Scaling Data
#####################-----------------------------------

# Objective: Preprocess and scale skewed and non-skewed data columns separately for better analysis.
def scale_data_skew_handling(df_pca, right_skewed_columns, not_skewed_columns):
    # Scaling skewed data
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    X_scaled_skewed = power_transformer.fit_transform(df_pca[right_skewed_columns])
    
    # Scaling non-skewed data
    scaler = StandardScaler()
    X_scaled_stand = scaler.fit_transform(df_pca[not_skewed_columns])
    
    # Create DataFrames
    X_scaled_skewed_df = pd.DataFrame(X_scaled_skewed, columns=right_skewed_columns)
    X_scaled_standard = pd.DataFrame(X_scaled_stand, columns=not_skewed_columns)
    
    # Concatenate the results
    X_scaled_df = pd.concat([X_scaled_skewed_df.reset_index(drop=True), X_scaled_standard.reset_index(drop=True)], axis=1)
    
    return X_scaled_df


#-----------------------------------------------------------------------------------------------

#####################-----------------------------------
##  PERFORMING PCA
#####################-----------------------------------

# Objective: PCA on the scaled data, and obtain principal components and their loadings.
def perform_pca(X_scaled_df, variance_threshold=0.90):
    # Perform PCA
    pca = PCA(variance_threshold)  # Use n_components to keep the desired variance
    X_pca = pca.fit_transform(X_scaled_df)
    
    # Analyze the explained variance by each component
    explained_variance = pca.explained_variance_ratio_
    print(f'Explained Variance: {explained_variance}')
    
    # Get the principal components
    principal_components = pca.components_
    
    # Create a DataFrame with the transformed data
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    
    # Obtain the loadings of the principal components
    loadings = pd.DataFrame(pca.components_.T, 
                            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                            index=X_scaled_df.columns)
    
    return pca_df, explained_variance, loadings


#-----------------------------------------------------------------------------------------------

#####################-----------------------------------
##  OPTIMAL K 
#####################-----------------------------------

# Objective: Determine the optimal number of clusters using the elbow plot and PCA plot
def cluster_elbow_plot_and_pca_plot(pca_df, k_range=range(1, 11)):
    # Convert pca_df to numpy array
    X_pca = pca_df.values

    # Define the range of k values
    k_values = k_range
    
    # Store results
    wcss = []  # List to hold the WCSS values
    
    for k in k_values:
        # Perform KMeans clustering on the PCA result
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        
        # Calculate WCSS and append to the list
        wcss.append(kmeans.inertia_)  # Inertia is the WCSS
    
    # Find the k value with the lowest WCSS
    min_wcss_k = k_values[wcss.index(min(wcss))]
    
    # Elbow Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss, marker='o', label='WCSS values')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.xticks(k_values)
    plt.grid()
    plt.legend()
    plt.show()
    
     # Plotting the PCA results for different k values
    plt.figure(figsize=(15, 10))
    for k in k_values[1:]:  # Start from k=2 to avoid empty plot for k=1
        # Re-run KMeans to get labels for plotting
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        plt.subplot(3, 4, k-1)  # Create subplots for each k (k-1 to start from index 0)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        plt.title(f'KMeans Clustering with k={k}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
    plt.tight_layout()
    plt.show()

    
    
#-----------------------------------------------------------------------------------------------

#####################-----------------------------------
##  Fitting K-Means
#####################-----------------------------------

# Objective: Fit the k-means and append clusters labels into the original data
def fitting_clusters(X_scaled, df, n_clusters= 3):
    # Fit the K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit_predict(X_scaled)
    
    # Assign cluster labels
    clusters = kmeans.labels_
    
    # Add cluster labels to the original data and PCA data
    df['Cluster'] = clusters
    
    return df


#-----------------------------------------------------------------------------------------------

#####################-----------------------------------
##  Kruskall-Wallis Test
#####################-----------------------------------

# Objective: Perform the Kruskal-Wallis test to determine if there are significant differences between clusters.
def kruskal_wallis_clusters(dataframe, cluster_column):
    
    # Initialize a list to store results
    results = []

    # Iterate through each column in the dataframe
    for column in dataframe.columns:
        if column != cluster_column:
            # Perform Kruskal-Wallis test
            groups = [group[column].dropna().values for name, group in dataframe.groupby(cluster_column)]
            stat, p = stats.kruskal(*groups)
            
            # Append results
            results.append({'Variable': column, 'H-statistic': stat, 'p-value': p})

            # Interpretation
            if p < 0.05:
                interpretation = f"There are significant differences between clusters for {column} (reject H0)."
            else:
                interpretation = f"There are no significant differences between clusters for {column} (do not reject H0)."
            
            print(f'Variable: {column},\n H-statistic: {stat},\n p-value: {p},\n Interpretation: {interpretation}')
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


#-----------------------------------------------------------------------------------------------

#####################-----------------------------------
##  Priority Segmentation
#####################-----------------------------------

# Objective: Classify counties within the target cluster into three priority categories using quantiles, 
# based on the population's low-income and low access to food.
def priority_classification(group):
    
    # Check if there are enough values to calculate quantiles
    if len(group['PCT_LACCESS_LOWI15']) < 3:  # At least three values needed for three categories
        group['Priority_Category'] = None  # Assign None if not enough valid data
    else:
        quantiles = group['PCT_LACCESS_LOWI15'].quantile([0.33, 0.67]).values
        group['Priority_Category'] = pd.cut(group['PCT_LACCESS_LOWI15'], 
                                   bins=[-float('inf'), quantiles[0], quantiles[1], float('inf')], 
                                   labels=['Low', 'Medium', 'High'])
    
    return group