# ==================================================
# File: multilingual_visualizations.py
# Visualisations multilingues
# ==================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from multilingual_config import MultilingualConfig

class MultilingualVisualizations:
    """Visualisations avec support multilingue"""
    
    def __init__(self, ml_config: MultilingualConfig):
        self.ml_config = ml_config
    
    def create_continental_overview(self, df: pd.DataFrame):
        """Vue continentale multilingue"""
        
        if df.empty:
            st.error(self.ml_config.t("no_data"))
            return
        
        # Calculs des métriques continentales
        from analytics import DemographicAnalytics
        analytics = DemographicAnalytics()
        continental_metrics = analytics.calculate_continental_metrics(df)
        
        if 'error' in continental_metrics:
            st.error(f"❌ {continental_metrics['error']}")
            return
        
        # Mise en évidence de la population
        pop_millions = continental_metrics.get('total_population_millions', 0)
        if pop_millions > 0:
            pop_formatted = self.ml_config.format_number(pop_millions, 0)
            billion_formatted = self.ml_config.format_number(pop_millions/1000, 2)
            
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #4CAF50, #2E7D32); color: white; 
                       padding: 1rem; border-radius: 8px; text-align: center; font-size: 1.2rem; margin: 1rem 0;">
                🌍 <strong>{self.ml_config.t("population")} ({self.ml_config.t("world_bank_api")})</strong><br>
                <strong>{pop_formatted} {self.ml_config.t("million")}</strong> ({billion_formatted} {self.ml_config.t("billion")})<br>
                📊 {self.ml_config.t("population_calculation")}
            </div>
            ''', unsafe_allow_html=True)
        
        # Métriques clés avec formatage localisé
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pop_display = f"{self.ml_config.format_number(pop_millions, 0)}M" if pop_millions > 0 else "N/A"
            st.metric(f"🌍 {self.ml_config.t('population')}", pop_display)
        
        with col2:
            median_age = continental_metrics.get('weighted_median_age', float('nan'))
            if not pd.isna(median_age):
                age_display = f"{self.ml_config.format_number(median_age, 1)} {self.ml_config.t('years')}"
            else:
                age_display = "N/A"
            st.metric(f"👥 {self.ml_config.t('median_age')}", age_display)
        
        with col3:
            tfr = continental_metrics.get('weighted_tfr', float('nan'))
            if not pd.isna(tfr):
                tfr_display = f"{self.ml_config.format_number(tfr, 1)}"
            else:
                tfr_display = "N/A"
            st.metric(f"👶 {self.ml_config.t('fertility_rate')}", tfr_display)
        
        with col4:
            growth_rate = continental_metrics.get('weighted_growth_rate', float('nan'))
            if not pd.isna(growth_rate):
                growth_display = f"{self.ml_config.format_number(growth_rate, 1)}{self.ml_config.t('percent_per_year')}"
            else:
                growth_display = "N/A"
            st.metric(f"📈 {self.ml_config.t('growth_rate')}", growth_display)
        
        # Tracker du dividende démographique multilingue
        st.markdown(f"### 🎯 {self.ml_config.t('demographic_dividend')} - {self.ml_config.t('real_time_data')}")
        self.create_dividend_tracker(continental_metrics.get('dividend_distribution', {}))
        
        # Carte interactive
        st.markdown(f"### 🗺️ {self.ml_config.t('select_indicator')}")
        
        available_indicators = [col for col in df.columns 
                              if col in ['total_fertility_rate', 'population_growth_rate', 'median_age', 'dividend_score']]
        
        if available_indicators:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_indicator = st.selectbox(
                    f"{self.ml_config.t('select_indicator')}:",
                    available_indicators,
                    format_func=lambda x: self.ml_config.get_indicator_name(x)
                )
            
            with col2:
                year_options = sorted(df['year'].unique(), reverse=True)
                selected_year = st.selectbox(f"{self.ml_config.t('select_year')}:", year_options)
            
            self.create_africa_map(df, selected_indicator, selected_year)
    
    def create_dividend_tracker(self, dividend_dist: dict):
        """Tracker dividende démographique multilingue"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        status_configs = [
            ("high_opportunity", "🟢", "#00C851", "#007E33"),
            ("opening_window", "🟡", "#ffbb33", "#FF8800"),
            ("limited_window", "🔴", "#ff4444", "#CC0000"),
            ("no_window", "⚪", "#6c757d", "#6c757d")
        ]
        
        for i, (col, (status_key, emoji, color1, color2)) in enumerate(zip([col1, col2, col3, col4], status_configs)):
            with col:
                count = dividend_dist.get(self.ml_config.get_text(status_key, "en"), 0)  # Les clés sont en anglais dans les données
                status_name = self.ml_config.t(status_key)
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color1}, {color2}); color: white; 
                           padding: 0.5rem; border-radius: 5px; text-align: center;">
                    <h3>{emoji} {status_name}</h3>
                    <h2>{count} pays</h2>
                    <p>{self.ml_config.t('demographic_dividend_desc')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def create_africa_map(self, df: pd.DataFrame, indicator: str, year: int):
        """Carte Afrique multilingue"""
        
        map_data = df[df['year'] == year].copy()
        if map_data.empty or indicator not in map_data.columns:
            st.error(f"{self.ml_config.t('no_data')} {self.ml_config.get_indicator_name(indicator)} {year}")
            return
        
        map_data = map_data.dropna(subset=[indicator])
        if map_data.empty:
            return
        
        # Mapping ISO2 vers ISO3
        iso2_to_iso3 = {
            'DZ': 'DZA', 'AO': 'AGO', 'BJ': 'BEN', 'BW': 'BWA', 'BF': 'BFA',
            'BI': 'BDI', 'CM': 'CMR', 'CV': 'CPV', 'CF': 'CAF', 'TD': 'TCD',
            'KM': 'COM', 'CG': 'COG', 'CD': 'COD', 'CI': 'CIV', 'DJ': 'DJI',
            'EG': 'EGY', 'GQ': 'GNQ', 'ER': 'ERI', 'SZ': 'SWZ', 'ET': 'ETH',
            'GA': 'GAB', 'GM': 'GMB', 'GH': 'GHA', 'GN': 'GIN', 'GW': 'GNB',
            'KE': 'KEN', 'LS': 'LSO', 'LR': 'LBR', 'LY': 'LBY', 'MG': 'MDG',
            'MW': 'MWI', 'ML': 'MLI', 'MR': 'MRT', 'MU': 'MUS', 'MA': 'MAR',
            'MZ': 'MOZ', 'NA': 'NAM', 'NE': 'NER', 'NG': 'NGA', 'RW': 'RWA',
            'ST': 'STP', 'SN': 'SEN', 'SC': 'SYC', 'SL': 'SLE', 'SO': 'SOM',
            'ZA': 'ZAF', 'SS': 'SSD', 'SD': 'SDN', 'TZ': 'TZA', 'TG': 'TGO',
            'TN': 'TUN', 'UG': 'UGA', 'ZM': 'ZMB', 'ZW': 'ZWE'
        }
        
        map_data['country_iso3'] = map_data['country_iso2'].map(iso2_to_iso3)
        map_data = map_data.dropna(subset=['country_iso3'])
        
        # Titre multilingue
        indicator_name = self.ml_config.get_indicator_name(indicator)
        title = f"{indicator_name} - {year}" if self.ml_config.get_language() == "fr" else f"Africa: {indicator_name} ({year})"
        
        fig = px.choropleth(
            map_data,
            locations='country_iso3',
            color=indicator,
            hover_name='country_name',
            color_continuous_scale='Viridis',
            title=title,
            labels={indicator: indicator_name}
        )
        
        fig.update_geos(
            projection_type="natural earth",
            showframe=False,
            showcoastlines=True,
            lonaxis_range=[-20, 55],
            lataxis_range=[-40, 40]
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_country_profiles(self, df: pd.DataFrame):
        """Profils pays multilingues"""
        st.markdown(f"### 🏛️ {self.ml_config.t('country_profiles')}")
        
        if 'country_name' not in df.columns:
            st.error(self.ml_config.t("no_data"))
            return
        
        available_countries = sorted(df['country_name'].unique())
        selected_country = st.selectbox(
            self.ml_config.t("select_country"), 
            available_countries
        )
        
        country_data = df[df['country_name'] == selected_country].copy()
        
        if not country_data.empty:
            latest_data = country_data[country_data['year'] == country_data['year'].max()].iloc[0]
            
            st.markdown(f"### 📊 {selected_country} - {latest_data['year']:.0f}")
            
            # Métriques avec formatage localisé
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'total_fertility_rate' in latest_data.index and pd.notna(latest_data['total_fertility_rate']):
                    tfr_val = self.ml_config.format_number(latest_data['total_fertility_rate'], 1)
                    st.metric(self.ml_config.t("fertility_rate"), tfr_val)
                else:
                    st.metric(self.ml_config.t("fertility_rate"), "N/A")
            
            with col2:
                if 'total_population' in latest_data.index and pd.notna(latest_data['total_population']):
                    pop_millions = latest_data['total_population'] / 1e6
                    pop_val = f"{self.ml_config.format_number(pop_millions, 1)}M"
                    st.metric(self.ml_config.t("population"), pop_val)
                else:
                    st.metric(self.ml_config.t("population"), "N/A")
            
            # Continue avec les autres métriques...
    
    def create_trend_analysis(self, df: pd.DataFrame):
        """Analyse tendances multilingue"""
        st.markdown(f"### 📈 {self.ml_config.t('trend_analysis')}")
        # Implémentation complète...
    
    def create_clustering_analysis(self, df: pd.DataFrame, analytics):
        """Analyse clustering multilingue"""
        st.markdown(f"### 🔬 {self.ml_config.t('clustering_analysis')}")
        # Implémentation complète...
    
    def create_data_explorer(self, df: pd.DataFrame):
        """Explorateur données multilingue"""
        st.markdown(f"### 🔍 {self.ml_config.t('data_explorer')}")
        # Implémentation complète...
    
    def create_api_status(self, api_service):
        """Statut API multilingue"""
        st.markdown(f"### 🔌 {self.ml_config.t('api_status')}")
        # Implémentation complète...
    
    def create_cache_management(self):
        """Gestion cache multilingue"""
        st.markdown(f"### 💾 {self.ml_config.t('cache_management')}")
        # Implémentation complète...

if __name__ == "__main__":
    main()