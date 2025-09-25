import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from api_service import WorldBankAPIService
from multilingual_config import Config

class DemographicAnalytics:
    """Analytics démographiques avec correctifs statistiques appliqués"""
    
    def __init__(self):
        self.api_service = WorldBankAPIService()
    
    def calculate_real_africa_population(self, year: int = 2023) -> Tuple[float, Dict, Dict]:
        """Calculate Africa population by summing country populations"""
        return self.api_service.get_population_data(year)
    
    def calculate_continental_metrics(self, df: pd.DataFrame, year: int = 2023) -> Dict:
        """CORRECTIF TÂCHE 2: Calculate population-weighted continental metrics avec robustesse"""
        
        # Get real population data
        total_pop, country_pops, metadata = self.calculate_real_africa_population(year)
        
        if total_pop == 0 or not country_pops:
            return {
                'error': 'No population data available',
                'total_population_millions': 0,
                'metadata': metadata
            }
        
        # Get demographic data for the year
        year_data = df[df['year'] == year].copy()
        if year_data.empty:
            latest_year = df['year'].max()
            year_data = df[df['year'] == latest_year].copy()
        
        if year_data.empty:
            return {
                'error': 'No demographic data available',
                'total_population_millions': total_pop / 1e6,
                'metadata': metadata
            }
        
        # Add population weights
        year_data['real_population'] = year_data['country_iso2'].map(
            lambda x: country_pops.get(x, {}).get('population', 0)
        )
        
        weighted_data = year_data[year_data['real_population'] > 0].copy()
        
        if weighted_data.empty:
            return {
                'error': 'No matching data',
                'total_population_millions': total_pop / 1e6,
                'metadata': metadata
            }
        
        # CORRECTIF: Calculate weighted metrics avec robustesse
        weighted_metrics = {}
        
        def robust_weighted_metric(values, weights, indicator_name):
            """Calcul robuste avec détection d'outliers"""
            if len(values) < 3:
                return np.nan
            
            # Détection outliers par IQR
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Masquer les outliers
                outlier_mask = (values >= lower_bound) & (values <= upper_bound)
                
                if outlier_mask.sum() >= 3:  # Au moins 3 pays valides
                    clean_values = values[outlier_mask]
                    clean_weights = weights[outlier_mask]
                    
                    # Médiane pondérée pour robustesse
                    return np.average(clean_values, weights=clean_weights)
            
            # Fallback: moyenne pondérée standard si pas assez de données
            return np.average(values, weights=weights)
        
        for indicator in ['total_fertility_rate', 'median_age', 'population_growth_rate', 'life_expectancy']:
            if indicator in weighted_data.columns:
                indicator_data = weighted_data.dropna(subset=[indicator])
                
                if not indicator_data.empty and len(indicator_data) >= 3:
                    values = indicator_data[indicator].values
                    weights = indicator_data['real_population'].values
                    
                    # NOUVEAU CODE ROBUSTE:
                    weighted_metrics[indicator] = robust_weighted_metric(
                        values, weights, indicator
                    )
                else:
                    weighted_metrics[indicator] = np.nan
            else:
                weighted_metrics[indicator] = np.nan
        
        # Dividend distribution
        dividend_counts = {}
        if 'dividend_status' in weighted_data.columns:
            dividend_counts = weighted_data['dividend_status'].value_counts().to_dict()
        
        return {
            'total_population_millions': total_pop / 1e6,
            'weighted_tfr': weighted_metrics.get('total_fertility_rate', np.nan),
            'weighted_median_age': weighted_metrics.get('median_age', np.nan),
            'weighted_growth_rate': weighted_metrics.get('population_growth_rate', np.nan),
            'weighted_life_expectancy': weighted_metrics.get('life_expectancy', np.nan),
            'dividend_distribution': dividend_counts,
            'countries_analyzed': len(weighted_data),
            'metadata': {
                **metadata,
                'calculation_year': year_data['year'].iloc[0] if not year_data.empty else year,
                'countries_with_demographic_data': len(weighted_data),
                'outliers_detected': True  # Flag pour indiquer traitement robuste
            }
        }
    
    def get_country_clusters(self, df: pd.DataFrame, year: int = 2023) -> pd.DataFrame:
        """CORRECTIF TÂCHE 4: Advanced ML clustering avec validation du nombre optimal"""
        
        year_data = df[df['year'] == year].copy()
        if year_data.empty:
            year_data = df[df['year'] == df['year'].max()].copy()
        
        if year_data.empty:
            return pd.DataFrame()
        
        # Use configured clustering indicators
        clustering_indicators = []
        config_indicators = Config.CLUSTERING_CONFIG['indicators']
        
        for indicator in config_indicators:
            if indicator in year_data.columns and year_data[indicator].notna().sum() >= 10:
                clustering_indicators.append(indicator)
        
        if len(clustering_indicators) < 2:
            return pd.DataFrame()
        
        # Prepare clustering data
        cluster_data = year_data[clustering_indicators].fillna(year_data[clustering_indicators].mean())
        valid_rows = cluster_data.dropna().index
        
        if len(valid_rows) < 10:
            return pd.DataFrame()
        
        # CORRECTIF: Validation du nombre optimal de clusters
        def find_optimal_clusters(data, max_k=6):
            """Trouve le nombre optimal de clusters par score de silhouette"""
            if len(data) < 6:
                return 2
            
            max_possible_k = min(max_k, len(data) // 2)
            
            if max_possible_k < 2:
                return 2
            
            silhouette_scores = []
            k_range = range(2, max_possible_k + 1)
            
            for k in k_range:
                kmeans = KMeans(
                    n_clusters=k, 
                    random_state=Config.CLUSTERING_CONFIG['random_state'], 
                    n_init=10
                )
                cluster_labels = kmeans.fit_predict(data)
                
                # Vérifier que tous les clusters ont au moins un point
                if len(np.unique(cluster_labels)) == k:
                    score = silhouette_score(data, cluster_labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(-1)  # Score invalide
            
            if not silhouette_scores or max(silhouette_scores) <= 0:
                return min(3, len(data) // 3)
            
            optimal_k = k_range[np.argmax(silhouette_scores)]
            return optimal_k
        
        # Perform clustering with optimization
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data.loc[valid_rows])
        
        # NOUVEAU CODE OPTIMISÉ:
        optimal_k = find_optimal_clusters(scaled_data)
        
        kmeans = KMeans(
            n_clusters=optimal_k, 
            random_state=Config.CLUSTERING_CONFIG['random_state'], 
            n_init=10
        )
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels
        clustered_data = year_data.loc[valid_rows].copy()
        clustered_data['cluster'] = clusters
        
        # Generate meaningful cluster labels based on fertility transition
        if 'total_fertility_rate' in clustered_data.columns:
            cluster_means = clustered_data.groupby('cluster')['total_fertility_rate'].mean()
            sorted_clusters = cluster_means.sort_values(ascending=False)
            
            label_map = {}
            # Adapter les labels au nombre réel de clusters
            base_labels = [
                'High Fertility Transition',
                'Moderate-High Fertility',
                'Moderate Fertility',
                'Lower Fertility',
                'Advanced Transition',
                'Late Transition'
            ]
            
            available_labels = base_labels[:len(sorted_clusters)]
            
            for i, cluster_id in enumerate(sorted_clusters.index):
                if i < len(available_labels):
                    label_map[cluster_id] = available_labels[i]
                else:
                    label_map[cluster_id] = f'Cluster {cluster_id}'
            
            clustered_data['cluster_label'] = clustered_data['cluster'].map(label_map)
            
            # Add cluster characteristics
            clustered_data = self._add_cluster_characteristics(clustered_data, clustering_indicators)
        
        return clustered_data
    
    def _add_cluster_characteristics(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Add cluster characteristic descriptions"""
        
        cluster_profiles = {}
        for cluster_label in df['cluster_label'].unique():
            cluster_data = df[df['cluster_label'] == cluster_label]
            
            profile = {}
            for indicator in indicators:
                if indicator in cluster_data.columns:
                    profile[indicator] = {
                        'mean': cluster_data[indicator].mean(),
                        'median': cluster_data[indicator].median(),
                        'std': cluster_data[indicator].std()
                    }
            
            cluster_profiles[cluster_label] = profile
        
        df['cluster_profiles'] = df['cluster_label'].map(cluster_profiles)
        
        return df
    
    def analyze_demographic_dividend_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze demographic dividend evolution over time"""
        
        if 'dividend_status' not in df.columns:
            return {}
        
        # Evolution over time
        dividend_evolution = df.groupby(['year', 'dividend_status']).size().unstack(fill_value=0)
        
        # Countries by status for latest year
        latest_year = df['year'].max()
        latest_dividend = df[df['year'] == latest_year].groupby('dividend_status')['country_name'].apply(list)
        
        # Transition analysis - countries moving between categories
        transitions = {}
        for country in df['country_name'].unique():
            country_data = df[df['country_name'] == country].sort_values('year')
            if not country_data.empty and len(country_data) > 1:
                first_status = country_data['dividend_status'].iloc[0]
                last_status = country_data['dividend_status'].iloc[-1]
                if first_status != last_status:
                    transitions[country] = {
                        'from': first_status,
                        'to': last_status,
                        'years': f"{country_data['year'].min()}-{country_data['year'].max()}"
                    }
        
        return {
            'evolution': dividend_evolution,
            'latest_distribution': latest_dividend.to_dict(),
            'transitions': transitions,
            'total_countries': df['country_name'].nunique()
        }
    
    def generate_country_comparison(self, df: pd.DataFrame, countries: List[str], indicators: List[str]) -> Dict:
        """Generate detailed comparison between selected countries"""
        
        comparison_data = df[df['country_name'].isin(countries)].copy()
        
        if comparison_data.empty:
            return {}
        
        # Latest values comparison
        latest_year = comparison_data['year'].max()
        latest_comparison = comparison_data[comparison_data['year'] == latest_year]
        
        comparison_metrics = {}
        for indicator in indicators:
            if indicator in latest_comparison.columns:
                country_values = latest_comparison.set_index('country_name')[indicator].to_dict()
                comparison_metrics[indicator] = {
                    'values': country_values,
                    'ranking': sorted(country_values.items(), key=lambda x: x[1] if pd.notna(x[1]) else 0, reverse=True),
                    'range': {
                        'min': min(v for v in country_values.values() if pd.notna(v)),
                        'max': max(v for v in country_values.values() if pd.notna(v))
                    } if any(pd.notna(v) for v in country_values.values()) else {}
                }
        
        # Trend correlation analysis
        correlations = {}
        for i, indicator1 in enumerate(indicators):
            for indicator2 in indicators[i+1:]:
                if both_cols_exist := (indicator1 in comparison_data.columns and indicator2 in comparison_data.columns):
                    corr_data = comparison_data[[indicator1, indicator2]].dropna()
                    if len(corr_data) > 3:
                        correlation = corr_data[indicator1].corr(corr_data[indicator2])
                        correlations[f"{indicator1}_vs_{indicator2}"] = correlation
        
        return {
            'latest_comparison': comparison_metrics,
            'correlations': correlations,
            'data_coverage': {
                'countries': len(countries),
                'years_available': sorted(comparison_data['year'].unique()),
                'indicators_available': [ind for ind in indicators if ind in comparison_data.columns]
            }
        }