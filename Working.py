# Enhanced NDVI-based Land Cover Classification Solution
# Hackathon: Summer Analytics Mid-Hackathon

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
df = pd.read_csv("hacktrain.csv")
test_data = pd.read_csv("hacktest.csv")

print(f"Training data shape: {df.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"\nClass distribution:")
print(df['class'].value_counts())

# Data exploration and visualization
def explore_data(df):
    """Explore the dataset and visualize key patterns"""
    print("\n=== DATA EXPLORATION ===")
    print(f"Missing values per column:")
    missing_values = df.isnull().sum()
    ndvi_cols = [col for col in df.columns if col.endswith('_N')]
    print(f"Total NDVI columns: {len(ndvi_cols)}")
    print(f"Missing values in NDVI columns: {missing_values[ndvi_cols].sum()}")
    
    # Plot missing values pattern
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(df[ndvi_cols[:10]].isnull(), cbar=True, yticklabels=False)
    plt.title('Missing Values Pattern (First 10 NDVI columns)')
    
    # Plot NDVI distribution by class
    plt.subplot(1, 2, 2)
    # Calculate mean NDVI per sample
    df_temp = df.copy()
    df_temp['mean_ndvi'] = df[ndvi_cols].mean(axis=1)
    sns.boxplot(data=df_temp, x='class', y='mean_ndvi')
    plt.xticks(rotation=45)
    plt.title('NDVI Distribution by Land Cover Class')
    plt.tight_layout()
    plt.show()

# Advanced preprocessing functions
class NDVIPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.imputer = KNNImputer(n_neighbors=5)
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def detect_outliers(self, X, contamination=0.1):
        """Detect outliers using IQR method"""
        outliers_mask = np.zeros(X.shape, dtype=bool)
        
        for i, col in enumerate(X.columns):
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask[:, i] = (X[col] < lower_bound) | (X[col] > upper_bound)
        
        return outliers_mask
    
    def create_temporal_features(self, X):
        """Create advanced temporal features from NDVI time series"""
        ndvi_cols = [col for col in X.columns if col.endswith('_N')]
        
        # Sort columns by date
        ndvi_cols_sorted = sorted(ndvi_cols, key=lambda x: x.split('_')[0])
        X_temporal = X[ndvi_cols_sorted].copy()
        
        features = pd.DataFrame(index=X.index)
        
        # Basic statistics
        features['ndvi_mean'] = X_temporal.mean(axis=1)
        features['ndvi_std'] = X_temporal.std(axis=1)
        features['ndvi_min'] = X_temporal.min(axis=1)
        features['ndvi_max'] = X_temporal.max(axis=1)
        features['ndvi_range'] = features['ndvi_max'] - features['ndvi_min']
        features['ndvi_median'] = X_temporal.median(axis=1)
        features['ndvi_skew'] = X_temporal.skew(axis=1)
        features['ndvi_kurtosis'] = X_temporal.kurtosis(axis=1)
        
        # Temporal patterns
        features['ndvi_trend'] = X_temporal.apply(lambda row: stats.linregress(
            range(len(row.dropna())), row.dropna())[0] if len(row.dropna()) > 1 else 0, axis=1)
        
        # Seasonal patterns (assuming data spans different seasons)
        features['ndvi_var_coeff'] = features['ndvi_std'] / (features['ndvi_mean'] + 1e-8)
        
        # Peak detection (maximum NDVI and its timing)
        features['ndvi_peak_value'] = X_temporal.max(axis=1)
        features['ndvi_peak_timing'] = X_temporal.idxmax(axis=1).apply(
            lambda x: ndvi_cols_sorted.index(x) if pd.notna(x) else -1)
        
        # Growing season metrics
        features['growing_season_length'] = (X_temporal > X_temporal.mean(axis=1).values.reshape(-1, 1)).sum(axis=1)
        
        return features
    
    def advanced_imputation(self, X):
        """Advanced imputation strategy for missing NDVI values"""
        X_imputed = X.copy()
        ndvi_cols = [col for col in X.columns if col.endswith('_N')]
        
        # For time series data, we can use forward/backward fill before KNN
        X_ndvi = X[ndvi_cols].copy()
        
        # Forward fill and backward fill
        X_ndvi = X_ndvi.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
        
        # Apply KNN imputation for remaining missing values
        if X_ndvi.isnull().sum().sum() > 0:
            X_ndvi_imputed = pd.DataFrame(
                self.imputer.fit_transform(X_ndvi),
                columns=ndvi_cols,
                index=X.index
            )
            X_imputed[ndvi_cols] = X_ndvi_imputed
        else:
            X_imputed[ndvi_cols] = X_ndvi
            
        return X_imputed
    
    def fit_transform(self, df, target_col='class'):
        """Fit preprocessor and transform training data"""
        # Separate features and target
        if target_col in df.columns:
            y = self.label_encoder.fit_transform(df[target_col])
            X = df.drop(columns=[target_col, 'ID'] if 'ID' in df.columns else [target_col])
        else:
            y = None
            X = df.drop(columns=['ID'] if 'ID' in df.columns else [])
        
        # Advanced imputation
        print("Applying advanced imputation...")
        X = self.advanced_imputation(X)
        
        # Create temporal features
        print("Creating temporal features...")
        temporal_features = self.create_temporal_features(X)
        
        # Combine original and temporal features
        X_combined = pd.concat([X, temporal_features], axis=1)
        
        # Handle any remaining missing values
        X_combined = X_combined.fillna(X_combined.mean())
        
        # Scale features
        print("Scaling features...")
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_combined),
            columns=X_combined.columns,
            index=X_combined.index
        )
        
        self.feature_names = X_scaled.columns.tolist()
        
        return X_scaled, y
    
    def transform(self, df):
        """Transform test data using fitted preprocessor"""
        X = df.drop(columns=['ID'] if 'ID' in df.columns else [])
        
        # Advanced imputation
        X = self.advanced_imputation(X)
        
        # Create temporal features
        temporal_features = self.create_temporal_features(X)
        
        # Combine features
        X_combined = pd.concat([X, temporal_features], axis=1)
        
        # Handle missing values
        X_combined = X_combined.fillna(X_combined.mean())
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_combined),
            columns=X_combined.columns,
            index=X_combined.index
        )
        
        return X_scaled

# Enhanced model training with hyperparameter tuning
class EnhancedLogisticRegression:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train_with_cv(self, X, y, cv_folds=5):
        """Train model with cross-validation and hyperparameter tuning"""
        print("\n=== MODEL TRAINING ===")
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'max_iter': [100, 500, 1000, 2000],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'penalty': ['l1', 'l2', 'elasticnet', 'none']
        }
        
        # Handle solver-penalty compatibility
        compatible_params = []
        for C in param_grid['C']:
            for max_iter in param_grid['max_iter']:
                for solver in param_grid['solver']:
                    for penalty in param_grid['penalty']:
                        if solver == 'lbfgs' and penalty in ['l2', 'none']:
                            compatible_params.append({'C': C, 'max_iter': max_iter, 'solver': solver, 'penalty': penalty})
                        elif solver == 'liblinear' and penalty in ['l1', 'l2']:
                            compatible_params.append({'C': C, 'max_iter': max_iter, 'solver': solver, 'penalty': penalty})
                        elif solver == 'saga' and penalty in ['l1', 'l2', 'elasticnet', 'none']:
                            compatible_params.append({'C': C, 'max_iter': max_iter, 'solver': solver, 'penalty': penalty})
        
        # Grid search with cross-validation
        base_model = LogisticRegression(multi_class='multinomial', random_state=42)
        
        print("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            base_model,
            compatible_params[:50],  # Limit search space for efficiency
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='accuracy')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)

def main():
    # Explore data
    explore_data(df)
    
    # Initialize preprocessor
    preprocessor = NDVIPreprocessor()
    
    # Preprocess training data
    print("\n=== PREPROCESSING ===")
    X_processed, y = preprocessor.fit_transform(df)
    
    print(f"Processed feature shape: {X_processed.shape}")
    print(f"Number of features: {len(preprocessor.feature_names)}")
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train enhanced model
    enhanced_model = EnhancedLogisticRegression()
    model = enhanced_model.train_with_cv(X_train, y_train)
    
    # Validation evaluation
    print("\n=== VALIDATION RESULTS ===")
    y_val_pred = enhanced_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_val, y_val_pred,
        target_names=preprocessor.label_encoder.classes_
    ))
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=preprocessor.label_encoder.classes_,
                yticklabels=preprocessor.label_encoder.classes_)
    plt.title('Confusion Matrix - Validation Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Feature importance analysis
    if hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': preprocessor.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(20), y='feature', x='importance')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.show()
    
    # Process test data and make predictions
    print("\n=== TEST DATA PREDICTION ===")
    test_ids = test_data['ID']
    X_test_processed = preprocessor.transform(test_data)
    
    # Make predictions
    y_test_pred = enhanced_model.predict(X_test_processed)
    y_test_pred_decoded = preprocessor.label_encoder.inverse_transform(y_test_pred)
    
    # Get prediction probabilities for confidence analysis
    y_test_proba = enhanced_model.predict_proba(X_test_processed)
    prediction_confidence = np.max(y_test_proba, axis=1)
    
    print(f"Test predictions shape: {y_test_pred.shape}")
    print(f"Mean prediction confidence: {prediction_confidence.mean():.4f}")
    print(f"Min prediction confidence: {prediction_confidence.min():.4f}")
    print(f"Prediction distribution:")
    unique, counts = np.unique(y_test_pred_decoded, return_counts=True)
    for class_name, count in zip(unique, counts):
        print(f"  {class_name}: {count}")
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': test_ids,
        'class': y_test_pred_decoded
    })
    
    submission.to_csv("enhanced_submission.csv", index=False)
    print("\nSubmission file saved as 'enhanced_submission.csv'")
    
    return submission, enhanced_model, preprocessor

# Run the enhanced solution
if __name__ == "__main__":
    submission, model, preprocessor = main()
    print("\n=== SOLUTION COMPLETE ===")
    print("Key improvements implemented:")
    print("1. Advanced imputation using KNN and temporal filling")
    print("2. Extensive temporal feature engineering")
    print("3. Robust scaling and outlier handling")
    print("4. Hyperparameter tuning with cross-validation")
    print("5. Comprehensive model evaluation")
    print("6. Feature importance analysis")