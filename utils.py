import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for better visualizations
import pandas as pd

def box_plot_viz(plt_data):
     plt.figure(figsize=(20,10))
     fig, ax = plt.subplots(figsize=(10, 7))
     bp = ax.boxplot(plt_data, labels=plt_data.columns,
               vert=True,
                    patch_artist=True, # Fill boxes with color
                    medianprops=dict(color='black', linewidth=2), # Median line color and width
                    boxprops=dict(facecolor='lightblue', edgecolor='blue', linewidth=1.5), # Box color and edge
                    whiskerprops=dict(color='gray', linewidth=1.5), # Whisker color and width
                    capprops=dict(color='gray', linewidth=1.5), # Cap color and width
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=8, linestyle='none', markeredgecolor='black') # Outlier properties
                    )
     ax.set_title('Comparison of Data Distributions using Box Plots', fontsize=16)
     ax.set_xlabel('Dataset', fontsize=12)
     ax.set_ylabel('Value', fontsize=12)
     ax.grid(axis='y', linestyle='--', alpha=0.7) # Add a grid for better readability
    #  plt.tight_layout() # Adjust layout to prevent labels from overlapping
     ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
     plt.show()


def violin_plot_viz(plt_data):
    plt.figure(figsize=(20, 10))
    melted_data = plt_data.melt(var_name='Dataset', value_name='Value')
    
    sns.violinplot(x='Dataset', y='Value', data=melted_data, palette='Set2', cut=0, linewidth=1.5)

    plt.title('Comparison of Data Distributions using Violin Plots', fontsize=16)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def line_plot_viz(X_columns, Y_columns, data):
    features = X_columns.copy()
    target = Y_columns.copy()
    df = data.copy()
    num_features = len(features)
    n_cols = 2 # You can adjust the number of columns in the grid
    n_rows = (num_features + n_cols - 1) // n_cols # Calculate rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    # 3. Iterate through each feature and plot its relationship with the target
    for i, feature in enumerate(features):
        ax = axes[i] # Get the current subplot axis

        # Create a scatter plot of the feature vs. the target
        sns.scatterplot(x=df[feature].values.flatten(), y=df[target].values.flatten(), ax=ax, alpha=0.7)

        # Add a regression line using seaborn's regplot (optional, but very informative)
        # This line shows the linear relationship that a simple linear regression would find
        sns.regplot(x=df[feature].values.flatten(), y=df[target].values.flatten(), ax=ax, scatter=False, color='red', line_kws={'linestyle':'--'}, label='Linear Trend')

        ax.set_title(f'{feature} vs. {target[0]}', fontsize=14)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel(target[0], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend() # Show the legend for the linear trend line
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # 4. Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.show()

def heat_map_viz(plt_data):
    plt.figure(figsize=(15, 8)) # Set the figure size for better readability

    # Use seaborn.heatmap to create the correlation heatmap
    sns.heatmap(
        plt_data.corr(),
        annot=True,      # Show the correlation values on the heatmap
        cmap='coolwarm', # Color map (coolwarm is good for correlations, diverging from 0)
        fmt=".2f",       # Format annotations to 2 decimal places
        linewidths=.5    # Add lines between cells
    )

    plt.title('Correlation Heatmap of All Variables', fontsize=16)
    plt.xticks(rotation=90, ha='right') # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)             # Keep y-axis labels horizontal
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()


def hist_plot_viz(columns, data):
    features = columns.copy()
    df = data.copy()
    num_features = len(features)
    n_cols = 3 # You can adjust the number of columns in the grid
    n_rows = (num_features + n_cols - 1) // n_cols # Calculate rows needed
    colors = ['r', 'g', 'b', 'm', 'c']
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    # 3. Iterate through each feature and plot its relationship with the target
    for i, feature in enumerate(features):
        ax = axes[i] # Get the current subplot axis

        sns.histplot(data[feature], kde=True, bins=30, ax=ax)

        ax.set_title(f'{feature}', fontsize=14)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # 4. Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.show()

def auto_corr_viz(processed_data):
    cols = list(processed_data.columns)
    num_features = len(cols)
    n_cols = 3 # You can adjust the number of columns in the grid
    n_rows = (num_features + n_cols - 1) // n_cols # Calculate rows needed
    colors = ['r', 'g', 'b', 'm', 'c']
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration
    i = 0
    for feature in cols:
        ax = axes[i] 
        target_df = pd.to_numeric(processed_data[feature], downcast='float')
        ax.acorr(target_df - target_df.mean(), maxlags=12, usevlines=True, normed=True)
        ax.set_title(f'{feature}', fontsize=14)
        ax.set_xlabel('Lag', fontsize=12)
        ax.set_ylabel('Autocorrelation', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        i = i + 1
    for j in range(i, len(axes)):
            fig.delaxes(axes[j])
    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.show()  