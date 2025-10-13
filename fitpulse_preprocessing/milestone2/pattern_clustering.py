"""
Pattern Clustering Module
Behavioral pattern discovery using machine learning clustering
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import streamlit as st

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class BehaviorPatternAnalyzer:
    """
    Advanced behavioral pattern analysis using clustering algorithms
    Supports K-Means, DBSCAN, and Hierarchical clustering
    """
    
    def __init__(self, scaling_method: str = 'standard'):
        """
        Initialize pattern analyzer
        
        Args:
            scaling_method: 'standard' or 'robust'
        """
        self.scaling_method = scaling_method
        self.feature_scalers = {}
        self.clustering_models = {}
        self.cluster_assignments = {}
        self.dimensionality_reducers = {}
        self.analysis_reports = {}
        
    def analyze_patterns(self, features_dict: Dict, method: str = 'kmeans',
                        n_clusters: int = 4, **kwargs) -> Dict:
        """
        Analyze patterns across all metrics
        
        Args:
            features_dict: Dictionary of feature matrices
            method: Clustering method
            n_clusters: Number of clusters
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing clustering results
        """
        results = {}
        
        for metric_name, feature_data in features_dict.items():
            if 'features' not in feature_data:
                continue
            
            feature_matrix = feature_data['features']
            
            st.subheader(f"Analyzing {metric_name.title()} Patterns")
            
            labels, report = self.cluster_metric_patterns(
                feature_matrix,
                metric_name,
                method=method,
                n_clusters=n_clusters,
                **kwargs
            )
            
            if len(labels) > 0:
                results[metric_name] = {
                    'labels': labels,
                    'report': report,
                    'model': self.clustering_models.get(metric_name)
                }
        
        return results
    
    def cluster_metric_patterns(self, feature_matrix: pd.DataFrame, metric_name: str,
                               method: str = 'kmeans', n_clusters: int = 4,
                               eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        Cluster patterns for a single metric
        
        Args:
            feature_matrix: Feature matrix from TSFresh
            metric_name: Metric identifier
            method: Clustering algorithm
            n_clusters: Number of clusters (for kmeans/hierarchical)
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples
            
        Returns:
            Cluster labels and analysis report
        """
        st.info(f"🔄 Applying {method.upper()} clustering...")
        
        report = {
            'metric': metric_name,
            'method': method,
            'n_samples': len(feature_matrix),
            'n_features': len(feature_matrix.columns)
        }
        
        try:
            if feature_matrix.empty:
                st.warning("Empty feature matrix provided")
                return np.array([]), report
            
            # Scale features
            scaled_features = self._scale_features(feature_matrix, metric_name)
            
            # Apply clustering algorithm
            if method.lower() == 'kmeans':
                labels, model_info = self._apply_kmeans_clustering(
                    scaled_features, n_clusters
                )
            elif method.lower() == 'dbscan':
                labels, model_info = self._apply_dbscan_clustering(
                    scaled_features, eps, min_samples
                )
            elif method.lower() == 'hierarchical':
                labels, model_info = self._apply_hierarchical_clustering(
                    scaled_features, n_clusters
                )
            else:
                st.error(f"Unknown clustering method: {method}")
                return np.array([]), report
            
            # Update report with model info
            report.update(model_info)
            
            # Calculate clustering quality metrics
            unique_labels = np.unique(labels)
            n_clusters_found = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
            
            if n_clusters_found > 1:
                # Silhouette Score: Measures how similar objects are to their own cluster
                # Range: -1 to 1, higher is better
                report['silhouette_score'] = silhouette_score(scaled_features, labels)
                
                # Davies-Bouldin Index: Average similarity ratio of each cluster with its most similar cluster
                # Range: 0 to infinity, lower is better
                report['davies_bouldin_index'] = davies_bouldin_score(scaled_features, labels)
                
                # Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion
                # Higher values indicate better-defined clusters
                report['calinski_harabasz_score'] = calinski_harabasz_score(scaled_features, labels)
            
            # Cluster distribution
            unique, counts = np.unique(labels, return_counts=True)
            report['cluster_distribution'] = {
                int(label): int(count) for label, count in zip(unique, counts)
            }
            report['n_clusters_found'] = n_clusters_found
            report['status'] = 'success'
            
            # Store results
            self.cluster_assignments[metric_name] = labels
            self.analysis_reports[metric_name] = report
            
            st.success(f"✅ Discovered {n_clusters_found} distinct behavioral patterns")
            
            # Display quality metrics
            if n_clusters_found > 1:
                col1, col2, col3 = st.columns(3)
                col1.metric("Silhouette", f"{report['silhouette_score']:.3f}")
                col2.metric("DB Index", f"{report['davies_bouldin_index']:.3f}")
                col3.metric("CH Score", f"{report['calinski_harabasz_score']:.1f}")
            
            return labels, report
            
        except Exception as e:
            report['status'] = 'failed'
            report['error'] = str(e)
            st.error(f"❌ Clustering failed: {str(e)}")
            return np.array([]), report
    
    def _scale_features(self, feature_matrix: pd.DataFrame, metric_name: str) -> np.ndarray:
        """
        Scale features using StandardScaler or RobustScaler
        
        Args:
            feature_matrix: Raw feature matrix
            metric_name: Metric identifier
            
        Returns:
            Scaled feature array
        """
        if self.scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        scaled_features = scaler.fit_transform(feature_matrix)
        self.feature_scalers[metric_name] = scaler
        
        return scaled_features
    
    def _apply_kmeans_clustering(self, features: np.ndarray, 
                                 n_clusters: int) -> Tuple[np.ndarray, Dict]:
        """Apply K-Means clustering"""
        
        model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=15,
            max_iter=500,
            random_state=42
        )
        
        labels = model.fit_predict(features)
        
        info = {
            'algorithm': 'K-Means',
            'n_clusters_requested': n_clusters,
            'inertia': float(model.inertia_),
            'n_iterations': int(model.n_iter_)
        }
        
        return labels, info
    
    def _apply_dbscan_clustering(self, features: np.ndarray, 
                                  eps: float, min_samples: int) -> Tuple[np.ndarray, Dict]:
        """Apply DBSCAN clustering"""
        
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        labels = model.fit_predict(features)
        
        n_noise = np.sum(labels == -1)
        
        info = {
            'algorithm': 'DBSCAN',
            'eps': eps,
            'min_samples': min_samples,
            'n_noise_points': int(n_noise),
            'noise_percentage': float((n_noise / len(labels)) * 100)
        }
        
        return labels, info
    
    def _apply_hierarchical_clustering(self, features: np.ndarray,
                                       n_clusters: int) -> Tuple[np.ndarray, Dict]:
        """Apply Hierarchical (Agglomerative) clustering"""
        
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        
        labels = model.fit_predict(features)
        
        info = {
            'algorithm': 'Hierarchical',
            'n_clusters_requested': n_clusters,
            'linkage_method': 'ward'
        }
        
        return labels, info
    
    def reduce_dimensions_and_visualize(self, feature_matrix: pd.DataFrame,
                                       metric_name: str, method: str = 'pca',
                                       n_components: int = 2):
        """
        Reduce dimensions for visualization
        
        Args:
            feature_matrix: Feature matrix
            metric_name: Metric identifier
            method: 'pca' or 'tsne'
            n_components: Number of dimensions
        """
        if metric_name not in self.cluster_assignments:
            st.warning("No clustering results available")
            return None
        
        labels = self.cluster_assignments[metric_name]
        scaler = self.feature_scalers.get(metric_name)
        
        if scaler is None:
            st.warning("Feature scaler not found")
            return None
        
        # Scale features
        scaled_features = scaler.transform(feature_matrix)
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_features = reducer.fit_transform(scaled_features)
            
            explained_var = reducer.explained_variance_ratio_
            st.info(
                f"PCA Explained Variance: "
                f"PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%}"
            )
        
        elif method.lower() == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(30, len(scaled_features) - 1),
                random_state=42,
                n_iter=1000
            )
            reduced_features = reducer.fit_transform(scaled_features)
        
        else:
            st.error(f"Unknown method: {method}")
            return None
        
        # Store reducer
        self.dimensionality_reducers[metric_name] = {
            'method': method,
            'reducer': reducer,
            'features_2d': reduced_features
        }
        
        return reduced_features, labels
    
    def get_cluster_profiles(self, feature_matrix: pd.DataFrame,
                            metric_name: str) -> Optional[pd.DataFrame]:
        """
        Generate cluster profiles (centroids)
        
        Args:
            feature_matrix: Original feature matrix
            metric_name: Metric identifier
            
        Returns:
            DataFrame with cluster profiles
        """
        if metric_name not in self.cluster_assignments:
            return None
        
        labels = self.cluster_assignments[metric_name]
        
        # Calculate cluster centroids
        cluster_profiles = []
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points
                continue
            
            cluster_mask = labels == cluster_id
            cluster_data = feature_matrix[cluster_mask]
            
            profile = {
                'Cluster': int(cluster_id),
                'Size': int(cluster_mask.sum()),
                'Mean': cluster_data.mean().mean(),
                'Std': cluster_data.std().mean(),
                'Min': cluster_data.min().min(),
                'Max': cluster_data.max().max()
            }
            
            cluster_profiles.append(profile)
        
        return pd.DataFrame(cluster_profiles)
