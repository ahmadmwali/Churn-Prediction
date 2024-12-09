import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Gym environment
import gymnasium as gym
from gymnasium import spaces

# Sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score, 
    silhouette_score,
    precision_score,
    recall_score
)

# XGBoost
from xgboost import XGBClassifier

class ChurnDataProcessor:
    """
    A comprehensive class for processing and preparing churn modeling data
    """
    def __init__(self, file_path):
        """
        Initialize the data processor with the given file path

        Args:
            file_path (str): Path to the Churn Modelling CSV file
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

        # Encoders and scalers
        self.geography_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()

    def preprocess_data(self, test_size=0.3, oversample=False):
        """
        Preprocess the dataset for machine learning models

        Args:
            test_size (float): Proportion of data to use for testing
            oversample (bool): Whether to oversample the minority class

        Returns:
            tuple: Processed training and testing datasets
        """
        # Encode categorical features
        df = self.data.copy()
        df["Geography"] = self.geography_encoder.fit_transform(df["Geography"])
        df["Gender"] = self.gender_encoder.fit_transform(df["Gender"])

        # Drop irrelevant columns
        df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

        # Split into features and target
        X = df.drop(columns=["Exited"])
        y = df["Exited"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Oversample if requested
        if oversample:
            X_train, y_train = self._oversample(X_train, y_train)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def _oversample(self, X_train, y_train):
        """
        Oversample the minority class

        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training labels

        Returns:
            tuple: Oversampled features and labels
        """
        minority_class = y_train.value_counts().idxmin()
        minority_indices = y_train[y_train == minority_class].index
        oversample_count = y_train.value_counts().max() - y_train.value_counts().min()

        oversample_indices = np.random.choice(minority_indices, size=oversample_count, replace=True)

        # Append oversampled data
        X_train_oversampled = pd.concat([X_train, X_train.loc[oversample_indices]])
        y_train_oversampled = pd.concat([y_train, y_train.loc[oversample_indices]])

        # Shuffle
        combined = pd.concat([X_train_oversampled, y_train_oversampled], axis=1)
        combined_shuffled = combined.sample(frac=1, random_state=42)

        return combined_shuffled.drop(columns=['Exited']), combined_shuffled['Exited']

    def revert_test_df(self, X_test, y_test, test_preds):
        """
        Revert scaled test features to original form and add predictions

        Args:
            X_test (ndarray): Scaled test features
            y_test (Series): True labels
            test_preds (ndarray): Predicted labels

        Returns:
            DataFrame: Test data with original features and predictions
        """
        # Convert scaled features back to original
        X_test_original = self.scaler.inverse_transform(X_test)

        # Create DataFrame
        test_df = pd.DataFrame(X_test_original, columns=[
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ])

        # Decode categorical variables
        test_df['Geography'] = self.geography_encoder.inverse_transform(test_df['Geography'].round().astype(int))
        test_df['Gender'] = self.gender_encoder.inverse_transform(test_df['Gender'].round().astype(int))

        # Append labels and predictions
        test_df['True_Label'] = y_test.values
        test_df['XGB_Label'] = test_preds

        return test_df

class XGBoostChurnClassifier:
    """
    XGBoost Classifier for Churn Prediction
    """
    def __init__(self):
        """
        Initialize the XGBoost classifier with default parameters
        """
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train, X_test, y_test):
        """
        Train the XGBoost classifier using GridSearchCV with recall optimization

        Args:
            X_train (ndarray): Training features
            y_train (Series): Training labels
            X_test (ndarray): Test features
            y_test (Series): Test labels

        Returns:
            XGBClassifier: Best trained model
        """
        # Define hyperparameter grid optimized for recall
        param_grid = {
            'scale_pos_weight': [1, 2, 3, 5],  # Helps with class imbalance
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }

        # Initialize XGBoost classifier
        xgb_clf = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss' 
        )

        # Perform GridSearchCV with recall scoring
        grid_search = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_grid,
            scoring='recall',  # Optimize for recall
            cv=5,
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        # Store best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print("XGBoost Classification Report:")
        print(classification_report(y_test, y_pred))

        return self.model

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluate model performance with recall focus

        Args:
            X_train (ndarray): Training features
            y_train (Series): Training labels
            X_test (ndarray): Test features
            y_test (Series): Test labels

        Returns:
            tuple: Training and test recalls and other metrics
        """
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)

        train_recall = recall_score(y_train, train_preds)
        test_recall = recall_score(y_test, test_preds)
        train_precision = precision_score(y_train, train_preds)
        test_precision = precision_score(y_test, test_preds)
        train_f1 = f1_score(y_train, train_preds)
        test_f1 = f1_score(y_test, test_preds)

        print(f"Training Recall: {train_recall}")
        print(f"Test Recall: {test_recall}")
        print(f"Training Precision: {train_precision}")
        print(f"Test Precision: {test_precision}")
        print(f"Training F1 Score: {train_f1}")
        print(f"Test F1 Score: {test_f1}")

        return train_recall, test_recall, train_precision, test_precision, train_f1, test_f1    
    
class ChurnKMeansClustering:
    """
    KMeans Clustering with PCA for Churn Analysis
    """
    def __init__(self, n_clusters=4, n_pca_components=5):
        """
        Initialize KMeans clustering with PCA

        Args:
            n_clusters (int): Number of clusters to create
            n_pca_components (int): Number of PCA components
        """
        self.n_clusters = n_clusters
        self.n_pca_components = n_pca_components
        self.kmeans_model = None
        self.pca_model = None
        self.silhouette_avg = None

    def perform_clustering(self, X_train_scaled, y_train):
        """
        Perform PCA and KMeans clustering on the data

        Args:
            X_train_scaled (ndarray): Scaled training features
            y_train (Series): True labels for the training set

        Returns:
            tuple: PCA-transformed data, KMeans model, clustering results, and silhouette score
        """
        # Perform PCA
        self.pca_model = PCA(n_components=self.n_pca_components, random_state=42)
        X_train_pca = self.pca_model.fit_transform(X_train_scaled)

        # Initialize KMeans
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=100,
            algorithm='elkan'
        )

        # Fit the KMeans model
        cluster_labels = self.kmeans_model.fit_predict(X_train_pca)

        # Calculate silhouette score
        self.silhouette_avg = silhouette_score(X_train_pca, cluster_labels)
        print(f"Silhouette Score: {self.silhouette_avg}")

        return X_train_pca, cluster_labels

    def predict_test_labels(self, X_test_scaled):
        """
        Predict labels for test set using PCA and KMeans

        Args:
            X_test_scaled (ndarray): Scaled test features

        Returns:
            tuple: PCA components for test set and cluster labels
        """
        # Transform test set using PCA
        X_test_pca = self.pca_model.transform(X_test_scaled)

        # Predict clusters for test set
        kmeans_test_labels = self.kmeans_model.predict(X_test_pca)

        return X_test_pca, kmeans_test_labels
    
class ChurnEnvironment(gym.Env):
    def __init__(self, X, y, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Initialize the environment.
        - X: Features
        - y: Labels
        - alpha: Learning rate
        - gamma: Discount factor
        - epsilon: Exploration rate for epsilon-greedy strategy
        """
        super().__init__()
        
        self.X = X.values if hasattr(X, 'values') else np.array(X)
        self.y = y.values if hasattr(y, 'values') else np.array(y)
        
        self.num_samples = len(X)
        self.current_index = 0
        
        # Q-table with shape (num_samples, num_actions)
        self.q_table = np.zeros((self.num_samples, 2))
        
        # Learning parameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Class balance handling
        class_counts = np.bincount(self.y)
        self.class_weights = 1 / (class_counts / len(self.y))
        
        # Action space
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.X.shape[1],), dtype=np.float32)

    def step(self, action):
        """
        Take an action in the environment.
        """
        true_label = self.y[self.current_index]
        
        # Weighted reward to handle class imbalance
        reward = self.class_weights[true_label] if action == true_label else -self.class_weights[true_label]

        # Move to the next index
        self.current_index += 1
        terminated = self.current_index >= self.num_samples
        truncated = False

        # Prepare next observation
        next_observation = self.X[self.current_index] if not terminated else None

        return next_observation, reward, terminated, truncated, {}

    def reset(self):
        """
        Reset the environment for a new episode.
        """
        self.current_index = 0
        return self.X[self.current_index], {}

    def choose_action(self, observation):
        """
        Choose action using epsilon-greedy strategy.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()  # Explore
        else:
            # Exploit (choose action with highest Q-value)
            return np.argmax(self.q_table[self.current_index])

def train_rl_churn_model(X_train, y_train, X_test, y_test, num_episodes=200):
    """
    Train a Q-learning based churn prediction model.
    
    Returns:
    - Predictions
    - Accuracy
    """
    env = ChurnEnvironment(X_train, y_train)

    # Training loop with Q-learning
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False

        while not done:
            # Choose and take action
            action = env.choose_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)

            # Q-value update
            if not terminated:
                # Q-learning update rule
                current_q = env.q_table[env.current_index - 1][action]
                max_next_q = np.max(env.q_table[env.current_index])
                new_q = current_q + env.alpha * (reward + env.gamma * max_next_q - current_q)
                env.q_table[env.current_index - 1][action] = new_q

            # Update observation
            observation = next_observation

            # Check if episode is done
            done = terminated or truncated

    # Make predictions on test set
    predictions = []
    for i in range(len(X_test)):
        env.current_index = i
        action = np.argmax(env.q_table[i])
        predictions.append(action)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(classification_report(y_test, predictions))

    return predictions, accuracy