"""
Advanced Behavioral Pattern Clustering Engine
Comprehensive clustering analysis using KMeans, DBSCAN, and Gaussian Mixture Models

Features:
- Multiple clustering algorithms with auto-parameter tuning
- Advanced preprocessing with feature scaling and selection
- Multi-dimensional visualization (PCA, t-SNE, UMAP)
- Cluster quality metrics and validation
- Automatic cluster interpretation
- Export capabilities for cluster results
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.feature_selection import SelectKBest, f_classif
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehaviorPatternAnalyzer:
    """
    Advanced behavioral pattern clustering engine
    
    Attributes:
        clusters (dict): Clustering results per metric
        scalers (dict): Feature scalers per metric
        models (dict): Trained clustering models
        dimensionality_reducers (dict): Dimensionality reduction models
        cluster_quality_metrics (dict): Quality metrics per clustering
    """
    
    def __init__(self):
        """Initialize the pattern analyzer"""
        self.clusters = {}
        self.scalers = {}
        self.models = {}
        self.dimensionality_reducers = {}
        self.cluster_quality_metrics = {}
        self.analysis_history = []
        
        logger.info("BehaviorPatternAnalyzer initialized")
    
    def analyze_patterns(
        self,
        feature_dict,
        method='kmeans',
        n_clusters=3,
        scaling_method='standard',
        dimensionality_reduction='pca',
        auto_tune=True,
        feature_selection=False,
        n_features_select=None
    ):
        """
        Comprehensive behavioral pattern analysis
        
        Args:
            feature_dict (dict): Dictionary of feature matrices per metric
            method (str): Clustering method ('kmeans', 'dbscan', 'gmm', 'hierarchical')
            n_clusters (int): Number of clusters (for applicable methods)
            scaling_method (str): Feature scaling ('standard', 'minmax', 'robust')
            dimensionality_reduction (str): Reduction method ('pca', 'tsne', 'umap')
            auto_tune (bool): Automatically tune hyperparameters
            feature_selection (bool): Apply feature selection
            n_features_select (int): Number of features to select
        
        Returns:
            dict: Comprehensive clustering results per metric
        """
        
        logger.info(f"Starting pattern analysis with {method} clustering")
        logger.info(f"  Scaling: {scaling_method}, Reduction: {dimensionality_reduction}")
        
        results = {}
        analysis_stats = {
            'start_time': datetime.now(),
            'method': method,
            'n_clusters': n_clusters,
            'scaling_method': scaling_method,
            'metrics_processed': []
        }
        
        for metric_name, features in feature_dict.items():
            try:
                logger.info(f"🎯 Analyzing patterns for {metric_name}")
                logger.info(f"  Features shape: {features.shape}")
                
                # Validate input
                self._validate_features(features, metric_name)
                
                # Feature preprocessing
                processed_features = self._preprocess_features(
                    features, 
                    metric_name,
                    scaling_method=scaling_method,
                    feature_selection=feature_selection,
                    n_features_select=n_features_select
                )
                
                # Auto-tune hyperparameters if requested
                if auto_tune:
                    logger.info(f"  Auto-tuning hyperparameters...")
                    optimal_params = self._auto_tune_hyperparameters(
                        processed_features, method, n_clusters
                    )
                else:
                    optimal_params = {'n_clusters': n_clusters}
                
                # Apply clustering
                logger.info(f"  Applying {method} clustering...")
                labels, model = self._apply_clustering(
                    processed_features, method, optimal_params
                )
                
                # Calculate quality metrics
                quality_metrics = self._calculate_clustering_quality(
                    processed_features, labels
                )
                
                # Dimensionality reduction for visualization
                logger.info(f"  Applying dimensionality reduction...")
                visualization_data = self._apply_dimensionality_reduction(
                    processed_features, 
                    labels,
                    method=dimensionality_reduction
                )
                
                # Cluster interpretation and statistics
                cluster_stats = self._calculate_comprehensive_cluster_stats(
                    processed_features, labels, features.columns, metric_name
                )
                
                # Cluster stability analysis
                stability_score = self._assess_cluster_stability(
                    processed_features, method, optimal_params
                )
                
                # Store comprehensive results
                results[metric_name] = {
                    'labels': labels,
                    'features_original': features,
                    'features_processed': processed_features,
                    'model': model,
                    'scaler': self.scalers[metric_name],
                    'quality_metrics': quality_metrics,
                    'visualization_data': visualization_data,
                    'cluster_stats': cluster_stats,
                    'stability_score': stability_score,
                    'optimal_params': optimal_params,
                    'config': {
                        'method': method,
                        'scaling_method': scaling_method,
                        'dimensionality_reduction': dimensionality_reduction,
                        'n_original_features': features.shape[1],
                        'n_processed_features': processed_features.shape[1]
                    }
                }
                
                # Store models and scalers
                self.models[metric_name] = model
                self.cluster_quality_metrics[metric_name] = quality_metrics
                
                analysis_stats['metrics_processed'].append(metric_name)
                
                logger.info(f"  ✅ Analysis complete. Silhouette: {quality_metrics['silhouette_score']:.3f}, "
                          f"Stability: {stability_score:.3f}")
                
            except Exception as e:
                logger.error(f"  ❌ Error analyzing {metric_name}: {str(e)}")
                raise RuntimeError(f"Pattern analysis failed for {metric_name}: {str(e)}")
        
        # Finalize statistics
        analysis_stats['end_time'] = datetime.now()
        analysis_stats['duration'] = (
            analysis_stats['end_time'] - analysis_stats['start_time']
        ).total_seconds()
        
        self.analysis_history.append(analysis_stats)
        self.clusters = results
        
        logger.info(f"✅ Pattern analysis complete. Duration: {analysis_stats['duration']:.2f}s")
        
        return results
    
    def _validate_features(self, features, metric_name):
        """Validate feature matrix"""
        
        if features is None or features.empty:
            raise ValueError(f"Feature matrix for {metric_name} is empty")
        
        if features.shape[0] < 10:
            raise ValueError(f"Insufficient samples for clustering (min 10)")
        
        if features.shape[1] < 2:
            raise ValueError(f"Insufficient features for clustering (min 2)")
        
        # Check for NaN or infinite values
        if features.isnull().any().any():
            raise ValueError(f"Feature matrix contains NaN values")
        
        if np.isinf(features.values).any():
            raise ValueError(f"Feature matrix contains infinite values")
    
    def _preprocess_features(
        self, 
        features, 
        metric_name,
        scaling_method='standard',
        feature_selection=False,
        n_features_select=None
    ):
        """Comprehensive feature preprocessing"""
        
        # Feature selection
        if feature_selection:
            if n_features_select is None:
                n_features_select = min(50, features.shape[1])
            
            # Use dummy target for unsupervised feature selection
            dummy_target = np.random.randint(0, 3, size=features.shape[0])
            selector = SelectKBest(f_classif, k=n_features_select)
            features_selected = pd.DataFrame(
                selector.fit_transform(features, dummy_target),
                columns=features.columns[selector.get_support()],
                index=features.index
            )
            logger.info(f"  Feature selection: {features.shape[1]} → {features_selected.shape[1]}")
            features = features_selected
        
        # Feature scaling
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        features_scaled = scaler.fit_transform(features)
        self.scalers[metric_name] = scaler
        
        logger.info(f"  Applied {scaling_method} scaling")
        
        return features_scaled
    
    def _auto_tune_hyperparameters(self, features, method, base_n_clusters):
        """Automatically tune clustering hyperparameters"""
        
        best_params = {}
        best_score = -1
        
        if method == 'kmeans':
            # Test different k values
            k_range = range(max(2, base_n_clusters-2), min(10, base_n_clusters+3))
            
            for k in k_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features)
                    
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(features, labels)
                        if score > best_score:
                            best_score = score
                            best_params = {'n_clusters': k}
                except:
                    continue
        
        elif method == 'dbscan':
            # Test different eps and min_samples
            eps_range = np.arange(0.3, 2.0, 0.2)
            min_samples_range = [3, 5, 7, 10]
            
            for eps in eps_range:
                for min_samples in min_samples_range:
                    try:
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = dbscan.fit_predict(features)
                        
                        if len(np.unique(labels)) > 1 and -1 not in labels:
                            score = silhouette_score(features, labels)
                            if score > best_score:
                                best_score = score
                                best_params = {'eps': eps, 'min_samples': min_samples}
                    except:
                        continue
        
        elif method == 'gmm':
            # Test different n_components
            n_range = range(max(2, base_n_clusters-2), min(10, base_n_clusters+3))
            
            for n in n_range:
                try:
                    gmm = GaussianMixture(n_components=n, random_state=42)
                    labels = gmm.fit_predict(features)
                    
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(features, labels)
                        if score > best_score:
                            best_score = score
                            best_params = {'n_components': n}
                except:
                    continue
        
        # Fallback to default if no good parameters found
        if not best_params:
            if method == 'kmeans':
                best_params = {'n_clusters': base_n_clusters}
            elif method == 'dbscan':
                best_params = {'eps': 0.5, 'min_samples': 5}
            elif method == 'gmm':
                best_params = {'n_components': base_n_clusters}
        
        logger.info(f"  Auto-tuned parameters: {best_params}, Score: {best_score:.3f}")
        
        return best_params
    
    def _apply_clustering(self, features, method, params):
        """Apply specified clustering algorithm"""
        
        if method == 'kmeans':
            model = KMeans(
                n_clusters=params['n_clusters'],
                random_state=42,
                n_init=10,
                max_iter=300
            )
            labels = model.fit_predict(features)
        
        elif method == 'dbscan':
            model = DBSCAN(
                eps=params.get('eps', 0.5),
                min_samples=params.get('min_samples', 5)
            )
            labels = model.fit_predict(features)
        
        elif method == 'gmm':
            model = GaussianMixture(
                n_components=params.get('n_components', 3),
                random_state=42,
                covariance_type='full'
            )
            labels = model.fit_predict(features)
        
        elif method == 'hierarchical':
            model = AgglomerativeClustering(
                n_clusters=params.get('n_clusters', 3),
                linkage='ward'
            )
            labels = model.fit_predict(features)
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return labels, model
    
    def _calculate_clustering_quality(self, features, labels):
        """Calculate comprehensive clustering quality metrics"""
        
        metrics = {}
        
        # Basic validation
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return {
                'silhouette_score': 0,
                'davies_bouldin_score': float('inf'),
                'calinski_harabasz_score': 0,
                'n_clusters': n_clusters,
                'n_noise_points': 0
            }
        
        # Silhouette Score
        try:
            silhouette = silhouette_score(features, labels)
        except:
            silhouette = 0
        
        # Davies-Bouldin Index
        try:
            davies_bouldin = davies_bouldin_score(features, labels)
        except:
            davies_bouldin = float('inf')
        
        # Calinski-Harabasz Index
        try:
            calinski_harabasz = calinski_harabasz_score(features, labels)
        except:
            calinski_harabasz = 0
        
        # Noise points (for DBSCAN)
        n_noise = np.sum(labels == -1) if -1 in labels else 0
        
        metrics = {
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies_bouldin),
            'calinski_harabasz_score': float(calinski_harabasz),
            'n_clusters': int(n_clusters),
            'n_noise_points': int(n_noise),
            'noise_percentage': float(n_noise / len(labels) * 100)
        }
        
        return metrics
    
    def _apply_dimensionality_reduction(self, features, labels, method='pca'):
        """Apply dimensionality reduction for visualization"""
        
        reducers = {}
        reduced_data = {}
        
        # PCA
        if method in ['pca', 'all']:
            pca = PCA(n_components=2, random_state=42)
            pca_data = pca.fit_transform(features)
            reducers['pca'] = pca
            reduced_data['pca'] = pca_data
            reduced_data['pca_variance_explained'] = pca.explained_variance_ratio_
        
        # t-SNE
        if method in ['tsne', 'all'] and len(features) <= 1000:  # t-SNE is slow for large datasets
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            tsne_data = tsne.fit_transform(features)
            reducers['tsne'] = tsne
            reduced_data['tsne'] = tsne_data
        
        # Store reducers
        self.dimensionality_reducers[f"{method}_reducers"] = reducers
        
        return reduced_data
    
    def _calculate_comprehensive_cluster_stats(self, features, labels, feature_names, metric_name):
        """Calculate detailed statistics for each cluster"""
        
        unique_labels = np.unique(labels)
        stats = {}
        
        for label in unique_labels:
            if label == -1:  # DBSCAN noise points
                cluster_name = 'Noise'
            else:
                cluster_name = f'Cluster {label}'
            
            cluster_mask = (labels == label)
            cluster_points = features[cluster_mask]
            
            # Basic statistics
            cluster_stats = {
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(labels) * 100),
                'centroid': np.mean(cluster_points, axis=0).tolist(),
                'std': np.std(cluster_points, axis=0).tolist(),
                'feature_importance': {}
            }
            
            # Feature importance (variance within cluster)
            if len(feature_names) == len(cluster_stats['centroid']):
                for i, feature_name in enumerate(feature_names):
                    cluster_stats['feature_importance'][feature_name] = {
                        'mean': float(cluster_stats['centroid'][i]),
                        'std': float(cluster_stats['std'][i])
                    }
            
            stats[cluster_name] = cluster_stats
        
        return stats
    
    def _assess_cluster_stability(self, features, method, params, n_trials=5):
        """Assess clustering stability through multiple runs"""
        
        stability_scores = []
        
        for trial in range(n_trials):
            try:
                # Add small random noise to features
                noisy_features = features + np.random.normal(0, 0.01, features.shape)
                
                # Apply clustering
                labels_base, _ = self._apply_clustering(features, method, params)
                labels_noisy, _ = self._apply_clustering(noisy_features, method, params)
                
                # Calculate adjusted rand index
                if len(np.unique(labels_base)) > 1 and len(np.unique(labels_noisy)) > 1:
                    ari = adjusted_rand_score(labels_base, labels_noisy)
                    stability_scores.append(ari)
                    
            except:
                continue
        
        return float(np.mean(stability_scores)) if stability_scores else 0.0
    
    def get_cluster_interpretation(self, metric_name):
        """Generate human-readable cluster interpretations"""
        
        if metric_name not in self.clusters:
            return None
        
        cluster_data = self.clusters[metric_name]
        labels = cluster_data['labels']
        features_original = cluster_data['features_original']
        
        interpretations = {}
        
        # Get most important features (highest variance)
        feature_importance = features_original.var().sort_values(ascending=False)
        top_features = feature_importance.head(3).index.tolist()
        
        for label in np.unique(labels):
            if label == -1:
                continue
            
            cluster_mask = (labels == label)
            cluster_features = features_original[cluster_mask]
            
            # Analyze top features
            feature_analysis = {}
            for feature in top_features:
                mean_value = cluster_features[feature].mean()
                feature_analysis[feature] = mean_value
            
            # Generate interpretation based on metric type and feature values
            interpretation = self._generate_cluster_interpretation(
                metric_name, feature_analysis, cluster_features
            )
            
            interpretations[f'Cluster {label}'] = {
                'interpretation': interpretation,
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(labels) * 100),
                'key_features': feature_analysis
            }
        
        return interpretations
    
    def _generate_cluster_interpretation(self, metric_name, feature_analysis, cluster_data):
        """Generate cluster interpretation based on metric type and features"""
        
        # Default interpretation
        interpretation = f"Pattern with distinct characteristics"
        
        # Heart rate specific interpretations
        if 'heart_rate' in metric_name.lower():
            # Look for heart rate related features
            hr_features = [f for f in feature_analysis.keys() if 'mean' in f.lower()]
            
            if hr_features:
                mean_hr = feature_analysis[hr_features[0]]
                if mean_hr < 70:
                    interpretation = "Resting State (Low Heart Rate Pattern)"
                elif mean_hr < 100:
                    interpretation = "Light Activity (Moderate Heart Rate Pattern)"
                else:
                    interpretation = "Exercise State (High Heart Rate Pattern)"
        
        # Steps specific interpretations
        elif 'step' in metric_name.lower():
            step_features = [f for f in feature_analysis.keys() if 'mean' in f.lower()]
            
            if step_features:
                mean_steps = feature_analysis[step_features[0]]
                if mean_steps < 50:
                    interpretation = "Sedentary Behavior Pattern"
                elif mean_steps < 120:
                    interpretation = "Moderate Activity Pattern"
                else:
                    interpretation = "High Activity Pattern"
        
        return interpretation
    
    def get_clustering_summary(self, metric_name):
        """Get comprehensive clustering summary"""
        
        if metric_name not in self.clusters:
            return None
        
        cluster_data = self.clusters[metric_name]
        
        summary = {
            'metric_name': metric_name,
            'quality_metrics': cluster_data['quality_metrics'],
            'cluster_stats': cluster_data['cluster_stats'],
            'interpretations': self.get_cluster_interpretation(metric_name),
            'stability_score': cluster_data['stability_score'],
            'config': cluster_data['config']
        }
        
        return summary
    
    def export_clustering_results(self, metric_name, filepath):
        """Export clustering results to CSV"""
        
        if metric_name not in self.clusters:
            raise ValueError(f"Metric {metric_name} not found")
        
        cluster_data = self.clusters[metric_name]
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'cluster_label': cluster_data['labels']
        })
        
        # Add original features
        original_features = cluster_data['features_original']
        for col in original_features.columns:
            results_df[f'feature_{col}'] = original_features[col].values
        
        results_df.to_csv(filepath, index=False)
        logger.info(f"Clustering results exported to {filepath}")
    
    def get_analysis_history(self):
        """Get history of all clustering analyses"""
        return self.analysis_history
