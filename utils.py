# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, recall_score

class EDA:
    def __init__(self, path):
        self.df = pd.read_csv(path)

    def overview(self):
        """Print an overview of the dataset."""
        print("Dataset Shape:", self.df.shape)
        print("Dataset Types:")
        print(self.df.dtypes)

    def plot_feature_boxplots(self):
        """Plot boxplots for numerical features in subplots."""
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).drop(
        columns=['IsActiveMember', 'HasCrCard', 'Exited', 'CustomerId']
        ).columns
    
        # Calculate subplot grid dimensions
        n_cols = 3
        n_rows = (len(numerical_columns) + n_cols - 1) // n_cols
    
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        fig.suptitle('Boxplots of Numerical Features', fontsize=16)
    
        # Flatten axes for easier iteration
        axes = axes.flatten() if n_rows > 1 else axes
    
        # Plot boxplots
        for i, col in enumerate(numerical_columns):
            sns.boxplot(data=self.df, x=col, ax=axes[i], palette='viridis')
            axes[i].set_title(f"Boxplot for {col}")
            axes[i].set_xlabel(col)
    
        # Remove any unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
    
        plt.tight_layout()
        plt.show()

    def plot_target_distribution(self, target_column):
        """Visualize the distribution of the target variable."""
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.df, x=target_column, palette='viridis')
        plt.title(f"Distribution of Target Variable: {target_column}")
        plt.xlabel(target_column)
        plt.ylabel("Count")
        plt.show()

    def show_description(self):
        """Show a statistical summary of the dataset."""
        print("Statistical Description of the Dataset:")
        return self.df.describe()

    def show_missing_values(self):
        """Show missing values in the dataset."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("No missing values found in the dataset.")
        else:
            print("Missing Values:")
            print(missing)

    def explore_correlation(self):
        """Explore correlation among numerical features."""
        correlation_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def explore_high_cardinality_columns(self, threshold=100):
        """Identify columns with too many unique values."""
        high_card_cols = [col for col in self.df.columns if self.df[col].nunique() > threshold]
        if high_card_cols:
            print(f"Columns with more than {threshold} unique values:")
            for col in high_card_cols:
                print(f"{col}: {self.df[col].nunique()} unique values")
        else:
            print(f"No columns with more than {threshold} unique values.")

    def optimal_clusters_elbow(self, max_clusters=10):
        """
        Find the optimal number of clusters using the elbow method.
        """
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        data = self.df[numerical_columns].dropna()
        distortions = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, max_clusters + 1), distortions, marker='o', linestyle='--')
        plt.title("Elbow Method for Optimal Clusters")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Distortion")
        plt.show()

    def check_duplicates(self):
        """Check for duplicate rows in the dataset."""
        duplicates = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")

# 1. Prediction Accuracy Comparison
def prediction_accuracy_comparison(df):
    methods = ['XGBoost', 'Reinforcement']
    accuracies = [
        (df['XGB_Label'] == df['True_Label']).mean() * 100,
        (df['RL_Label'] == df['True_Label']).mean() * 100
    ]

    fig = px.bar(
        x=methods,
        y=accuracies,
        title='Prediction Accuracy Across Different Methods',
        labels={'x': 'Prediction Method', 'y': 'Accuracy (%)'}
    )
    return fig

# 2. Prediction F1 Score Comparison
def prediction_f1_comparison(df):
    methods = ['XGBoost', 'Reinforcement']

    # Compute F1 scores for each method
    f1_scores = [
        recall_score(df['True_Label'], df['XGB_Label']),
        recall_score(df['True_Label'], df['RL_Label'])
    ]

    # Plot F1 scores
    fig = px.bar(
        x=methods,
        y=f1_scores,
        title='Recall Score Comparison Across Different Methods',
        labels={'x': 'Prediction Method', 'y': 'Recall Score'},
        text=[f"{score:.2f}" for score in f1_scores]  # Show F1 scores on bars
    )

    fig.update_traces(textposition='outside')  # Display text outside bars
    return fig

# 3. Churn Rate Comparison
def churn_rate_comparison(df):
    churn_comparison = pd.DataFrame({
        'Prediction Method': ['True Label', 'XGBoost', 'Reinforcement'],
        'Churn Rate': [
            df['True_Label'].mean() * 100,
            df['XGB_Label'].mean() * 100,
            df['RL_Label'].mean() * 100
        ]
    })

    fig = px.bar(churn_comparison, x='Prediction Method', y='Churn Rate',
                 title='Churn Rate Comparison Across Different Methods',
                 labels={'Churn Rate': 'Churn Rate (%)'})
    return fig

# 4. Age Distribution by Churn Status
def age_distribution_by_churn(df):
    fig = px.box(df, x='XGB_Label', y='Age',
                 title='Age Distribution by Churn Status',
                 labels={'True_Label': 'Churn Status', 'Age': 'Age'})
    return fig

# 5. Credit Score Distribution
def credit_score_distribution(df):
    fig = px.histogram(df, x="CreditScore", color="XGB_Label",
                   title="Distribution of Credit Score by Churn",
                   labels={"Label": "Churn"}, barmode='group')
    return fig

# 6. Credit Score vs Estimated Salary
def credit_score_vs_salary(df):
    fig = px.scatter(df, x='CreditScore', y='EstimatedSalary', color='XGB_Label',
                     title='Credit Score vs Estimated Salary',
                     labels={'True_Label': 'Churn Status'})
    return fig

# 7. Balance Distribution by Churn
def balance_by_churn(df):
    fig = px.box(df, y='Balance', x='XGB_Label',
                 title='Balance Distribution by Churn',
                 labels={'True_Label': 'Churn Status', 'Balance': 'Account Balance'})
    return fig

# 8. Geographical Churn Distribution
def geographical_churn_distribution(df):
    geo_churn = df.groupby('Geography')['XGB_Label'].mean() * 100
    fig = px.bar(
        x=geo_churn.index,
        y=geo_churn.values,
        title='Churn Rate by Geography',
        labels={'x': 'Geography', 'y': 'Churn Rate (%)'}
    )
    return fig

# 9. Churn by Gender and Geography
def churn_by_gender_geography(df):
    churn_breakdown = df.groupby(['Geography', 'Gender', 'XGB_Label']).size().reset_index(name='Count')
    fig = px.treemap(churn_breakdown,
                     path=['Geography', 'Gender', 'XGB_Label'],
                     values='Count',
                     title='Churn Breakdown by Geography and Gender')
    return fig

# 10. Correlation Heatmap
def correlation_heatmap(df):
    numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                    'XGB_Label']
    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(corr_matrix,
                    title='Correlation Heatmap of Numeric Features',
                    color_continuous_scale='RdBu_r')
    return fig

# 11. Tenure Distribution by Churn
def tenure_distribution_by_churn(df):
    fig = px.box(df, x='XGB_Label', y='Tenure',
                 title='Tenure Distribution by Churn Status',
                 labels={'True_Label': 'Churn Status', 'Tenure': 'Tenure (Years)'})
    return fig

# 12. Active Membership Impact
def active_membership_impact(df):
    active_churn = df.groupby('IsActiveMember')['XGB_Label'].mean() * 100
    fig = px.bar(
        x=active_churn.index,
        y=active_churn.values,
        title='Churn Rate by Active Membership Status',
        labels={'x': 'Is Active Member', 'y': 'Churn Rate (%)'}
    )
    return fig

# 13. Credit Card Ownership Impact
def credit_card_impact(df):
    card_churn = df.groupby('HasCrCard')['XGB_Label'].mean() * 100
    fig = px.bar(
        x=card_churn.index,
        y=card_churn.values,
        title='Churn Rate by Credit Card Ownership',
        labels={'x': 'Has Credit Card', 'y': 'Churn Rate (%)'}
    )
    return fig

# 14. Salary Distribution by Churn
def salary_distribution_by_churn(df):
    fig = px.box(df, x='XGB_Label', y='EstimatedSalary',
                 title='Salary Distribution by Churn Status',
                 labels={'True_Label': 'Churn Status', 'EstimatedSalary': 'Estimated Salary'})
    return fig

# 15. PCA Scatter Plot
def pca_scatter(df):
    fig = px.scatter(
        df,
        x='PCA1',
        y='PCA2',
        color='KMeans_Label',
        title='Clusters on PCA Axes',
        labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
        template='plotly'
    )
    return fig

def create_dashboard(df):  # Add back the df parameter
    # Create subplots with different subplot types (xy for charts, treemap for treemaps)
    fig = make_subplots(
        rows=5, cols=3,  # Adjust the grid size as per the number of plots
        subplot_titles=[
            'Prediction Accuracy Scores Across Different Models',
            'Prediction Recall Scores Across Different Models',
            'Churn Rates Across Different Models',
            'Age Distribution by Churn Status',
            'Credit Score Distribution by Churn Status',
            'Credit Score vs Estimated Salary',
            'Balance Distribution by Churn', 
            'Churn Rate by Geography',		
            'Churn Breakdown by Geography and Gender',		 
            'Correlation Heatmap of Numeric Features',		
            'Tenure Distribution by Churn Status',         
            'Churn Rate by Active Membership Status',
            'Churn Rate by Credit Card Ownership',
            'Salary Distribution by Churn Status',
            'Visualization of Clusters on 2 PCA Axes'
        ],
        vertical_spacing=0.1, # Adjust spacing between subplots
        horizontal_spacing=0.1,
        specs=[
            [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}, {'type': 'treemap'}],
            [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]
        ]
    )

    # Add each plot to the corresponding subplot location
    fig.add_trace(prediction_accuracy_comparison(df).data[0], row=1, col=1)
    fig.add_trace(prediction_f1_comparison(df).data[0], row=1, col=2)
    fig.add_trace(churn_rate_comparison(df).data[0], row=1, col=3)
    fig.add_trace(age_distribution_by_churn(df).data[0], row=2, col=1)
    fig.add_trace(credit_score_distribution(df).data[0], row=2, col=2)
    fig.add_trace(credit_score_distribution(df).data[1], row=2, col=2)
    fig.add_trace(credit_score_vs_salary(df).data[0], row=2, col=3)
    fig.add_trace(balance_by_churn(df).data[0], row=3, col=1)
    fig.add_trace(geographical_churn_distribution(df).data[0], row=3, col=2)
    fig.add_trace(churn_by_gender_geography(df).data[0], row=3, col=3)
    fig.add_trace(correlation_heatmap(df).data[0], row=4, col=1)
    fig.add_trace(tenure_distribution_by_churn(df).data[0], row=4, col=2)
    fig.add_trace(active_membership_impact(df).data[0], row=4, col=3)
    fig.add_trace(credit_card_impact(df).data[0], row=5, col=1)
    fig.add_trace(salary_distribution_by_churn(df).data[0], row=5, col=2)
    fig.add_trace(pca_scatter(df).data[0], row=5, col=3)

    # Update layout
    fig.update_layout(
        height=2000,  # Adjust the height of the entire dashboard
        title_text='Churn Prediction Analysis Dashboard',
        showlegend=False,
        title_x=0.5  # Center the title
    )

    return fig
