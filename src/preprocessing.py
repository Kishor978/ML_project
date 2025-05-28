import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.corr_features_to_drop = None 
        self.pca = None
        self.var_thresh = VarianceThreshold(threshold=0.0)
        
    def preprocess_data(self, X, fit_transform=True):
        """Complete preprocessing pipeline"""
        print("Starting preprocessing...")
        
        X_clean = self._handle_missing_values(X)
        X_var = self._handle_variance(X_clean, fit_transform)
        X_corr = self._handle_correlation(X_var, fit_transform)
        X_scaled = self._scale_features(X_corr, fit_transform)
        X_pca = self._reduce_dimensions(X_scaled, fit_transform)
        
        print(f"Preprocessing complete. Final shape: {X_pca.shape}")
        return X_pca
    
    def _handle_missing_values(self, X):
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        missing_threshold = 0.9 * len(X_clean)
        X_clean = X_clean.dropna(axis=1, thresh=missing_threshold)
        return X_clean.fillna(X_clean.mean())
    
    def _handle_variance(self, X, fit_transform):
        if fit_transform:
            return self.var_thresh.fit_transform(X)
        return self.var_thresh.transform(X)
    
    def _handle_correlation(self, X, fit_transform, threshold=0.95):
        """Handle correlated features while preventing data leakage"""
        X_df = pd.DataFrame(X)
        
        if fit_transform:
            # Calculate correlations and identify features to drop only during training
            corr_matrix = X_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            self.corr_features_to_drop = [column for column in upper.columns 
                                        if any(upper[column] > threshold)]
            print(f"Identified {len(self.corr_features_to_drop)} highly correlated features to drop")
        
        # Use the same features identified during training
        if self.corr_features_to_drop:
            X_df = X_df.drop(columns=self.corr_features_to_drop)
            
        return X_df.values
    
    def _scale_features(self, X, fit_transform):
        if fit_transform:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def _reduce_dimensions(self, X, fit_transform):
        if fit_transform:
            self.pca = PCA(n_components=0.9, random_state=42)
            return self.pca.fit_transform(X)
        return self.pca.transform(X)