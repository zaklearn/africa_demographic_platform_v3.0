import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from multilingual_config import Config, MultilingualConfig
from api_service import WorldBankAPIService
from analytics import DemographicAnalytics
from cache_manager import CacheManager
from debug_tools import DebugTools
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config - MODE CLAIR PAR DÉFAUT
st.set_page_config(
    page_title="🌍 Africa Demographics Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS - MODE CLAIR + POLICE TABLEAUX AUGMENTÉE
st.markdown("""
<style>
    /* Force mode clair */
    .stApp {
        background-color: white !important;
        color: black !important;
    }
    
    /* Tableaux avec police augmentée */
    .dataframe {
        font-size: 16px !important;
    }
    
    .dataframe td, .dataframe th {
        font-size: 16px !important;
        padding: 8px !important;
    }
    
    /* Styles existants préservés */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .dividend-status {
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        padding: 0.5rem;
        background: linear-gradient(90deg, #f0f0f0, #e0e0e0);
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def load_demographic_data(use_core_only: bool = False):
    """Load and cache demographic data"""
    service = WorldBankAPIService()
    return service.load_all_demographic_data(use_core_only=use_core_only)

def create_multilingual_continental_overview(df: pd.DataFrame, ml_config: MultilingualConfig, analytics: DemographicAnalytics):
    """Vue continentale avec support multilingue"""
    
    if df.empty:
        st.error(ml_config.t("no_data"))
        return
    
    # Calculate continental metrics
    continental_metrics = analytics.calculate_continental_metrics(df)
    
    if 'error' in continental_metrics:
        st.error(f"❌ {continental_metrics['error']}")
        return
    
    # Population highlight - VALEURS ARRONDIES
    pop_millions = continental_metrics.get('total_population_millions', 0)
    if pop_millions > 0:
        pop_formatted = ml_config.format_number(pop_millions, 0)
        billion_formatted = ml_config.format_number(pop_millions/1000, 2)
        
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, #4CAF50, #2E7D32); color: white; 
                   padding: 1rem; border-radius: 8px; text-align: center; font-size: 1.2rem; margin: 1rem 0;">
            🌍 <strong>{ml_config.t("population")} ({ml_config.t("world_bank_api")})</strong><br>
            <strong>{pop_formatted} {ml_config.t("million")}</strong> ({billion_formatted} {ml_config.t("billion")})<br>
            📊 {ml_config.t("population_calculation")}
        </div>
        ''', unsafe_allow_html=True)
    
    # Key metrics - VALEURS ARRONDIES
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pop_display = f"{ml_config.format_number(pop_millions, 0)}M" if pop_millions > 0 else "N/A"
        st.metric(f"🌍 {ml_config.t('population')}", pop_display)
    
    with col2:
        median_age = continental_metrics.get('weighted_median_age', float('nan'))
        age_display = f"{ml_config.format_number(median_age, 1)} {ml_config.t('years')}" if not pd.isna(median_age) else "N/A"
        st.metric(f"👥 {ml_config.t('median_age')}", age_display)
    
    with col3:
        tfr = continental_metrics.get('weighted_tfr', float('nan'))
        tfr_display = f"{ml_config.format_number(tfr, 1)}" if not pd.isna(tfr) else "N/A"
        st.metric(f"👶 {ml_config.t('fertility_rate')}", tfr_display)
    
    with col4:
        growth_rate = continental_metrics.get('weighted_growth_rate', float('nan'))
        growth_display = f"{ml_config.format_number(growth_rate, 1)}{ml_config.t('percent_per_year')}" if not pd.isna(growth_rate) else "N/A"
        st.metric(f"📈 {ml_config.t('growth_rate')}", growth_display)
    
    # Demographic Dividend Tracker
    st.markdown(f"### 🎯 {ml_config.t('demographic_dividend')} - {ml_config.t('real_time_data')}")
    create_multilingual_dividend_tracker(continental_metrics.get('dividend_distribution', {}), ml_config)

def create_multilingual_dividend_tracker(dividend_dist: dict, ml_config: MultilingualConfig):
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
            english_status = ml_config.translator.get_text(status_key, "en")
            count = dividend_dist.get(english_status, 0)
            status_name = ml_config.t(status_key)
            countries_text = "pays" if ml_config.get_language() == "fr" else "countries"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color1}, {color2}); color: white; 
                       padding: 0.5rem; border-radius: 5px; text-align: center;">
                <h3>{emoji} {status_name}</h3>
                <h2>{count} {countries_text}</h2>
                <p>{ml_config.t('demographic_dividend_desc')}</p>
            </div>
            """, unsafe_allow_html=True)

def create_multilingual_africa_map(df: pd.DataFrame, indicator: str, year: int, ml_config: MultilingualConfig):
    """Carte Afrique avec représentation unifiée du Maroc"""
    
    # Récupération des données avec fallback
    map_data = get_best_available_data(df, indicator, year, max_years_back=3)
    
    if map_data.empty:
        st.warning(f"Aucune donnée disponible pour {indicator}")
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
    
    # SOLUTION: Dupliquer les données du Maroc pour le Sahara Occidental
    morocco_data = map_data[map_data['country_iso3'] == 'MAR']
    if not morocco_data.empty:
        # Créer entrée pour Sahara Occidental avec les mêmes données que le Maroc
        sahara_data = morocco_data.copy()
        sahara_data['country_iso3'] = 'ESH'
        sahara_data['country_name'] = 'Morocco'
        sahara_data['country_iso2'] = 'EH'
        
        # Ajouter aux données cartographiques
        map_data = pd.concat([map_data, sahara_data], ignore_index=True)
    
    # Configuration titre
    try:
        indicator_name = ml_config.translator.get_indicator_name(indicator, ml_config.get_language())
    except:
        indicator_name = indicator.replace('_', ' ').title()
    
    if ml_config.get_language() == "fr":
        title = f"Afrique: {indicator_name} ({year})"
    else:
        title = f"Africa: {indicator_name} ({year})"
    
    # Créer la carte
    fig = px.choropleth(
        map_data,
        locations='country_iso3',
        color=indicator,
        hover_name='country_name',
        hover_data={
            indicator: ':.2f',
            'year': True,
            'country_iso3': False
        },
        color_continuous_scale='Viridis',
        title=title,
        labels={indicator: indicator_name}
    )
    
    # Configuration géographique spécialisée
    fig.update_geos(
        projection_type="natural earth",
        showframe=False,
        showcoastlines=True,
        showcountries=True,
        countrycolor="rgba(128,128,128,0.3)",  # Frontières légères
        # Masquer certaines frontières sensibles
        visible=False,
        resolution=50,  # Résolution plus basse pour masquer détails
        lonaxis_range=[-25, 55],
        lataxis_range=[-40, 40],
        projection=dict(
            rotation=dict(lon=15, lat=0)
        )
    )
    
    fig.update_layout(
        height=600,
        title_x=0.5,
        coloraxis_colorbar=dict(
            title=indicator_name,
            title_side="right"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Note explicative
    if any(map_data['country_iso3'] == 'ESH'):
        st.caption("ℹ️ Les données du Maroc incluent l'ensemble du territoire selon la configuration géographique utilisée.")
    
    return fig

def get_best_available_data(df, indicator, target_year, max_years_back=3):
    """Helper pour récupérer données avec fallback"""
    result_data = []
    
    for country in df['country_iso2'].unique():
        country_data = df[df['country_iso2'] == country]
        
        for year_offset in range(max_years_back + 1):
            check_year = target_year - year_offset
            year_data = country_data[country_data['year'] == check_year]
            
            if not year_data.empty and pd.notna(year_data[indicator].iloc[0]):
                result_data.append({
                    'country_iso2': country,
                    'country_name': year_data['country_name'].iloc[0],
                    'year': check_year,
                    indicator: round(year_data[indicator].iloc[0], 2)
                })
                break
    
    return pd.DataFrame(result_data)

def create_population_pyramid(df: pd.DataFrame, country_name: str, year: int = 2023, animate: bool = False):
    """Pyramide population avec distribution réaliste et valeurs arrondies"""
    
    country_data = df[df['country_name'] == country_name].copy()
    
    if country_data.empty:
        st.error(f"No data available for {country_name}")
        return
    
    if animate:
        animation_years = sorted(country_data['year'].unique())
        pyramid_data = country_data
    else:
        pyramid_data = country_data[country_data['year'] == year]
        animation_years = [year]
    
    if pyramid_data.empty:
        st.error(f"No data for {country_name} in {year}")
        return
    
    age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', 
                  '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', 
                  '70-74', '75-79', '80+']
    
    fig = go.Figure()
    
    def generate_realistic_age_distribution(tfr, life_exp, growth_rate):
        """Distribution démographique réaliste avec valeurs arrondies"""
        
        tfr = np.clip(tfr if pd.notna(tfr) else 4.0, 1.5, 8.0)
        life_exp = np.clip(life_exp if pd.notna(life_exp) else 60, 40, 85)
        growth_rate = np.clip(growth_rate if pd.notna(growth_rate) else 2.5, -1, 5)
        
        base_population = 100000
        survival_rates = []
        
        for i, age_group in enumerate(age_groups):
            if i < 3:
                survival = 0.95 + (life_exp - 50) * 0.001
            elif i < 13:
                survival = 0.98 - (i - 3) * 0.005
            else:
                decline_factor = (85 - life_exp) * 0.01
                survival = max(0.3, 0.85 - (i - 13) * 0.1 - decline_factor)
            
            survival_rates.append(max(0.1, min(0.99, survival)))
        
        age_distribution = []
        
        for i, age_group in enumerate(age_groups):
            if i == 0:
                base_births = base_population * (tfr / 5.0) * 0.048
                population = base_births * survival_rates[i]
            else:
                prev_pop = age_distribution[i-1] if i > 0 else base_births
                growth_effect = (1 + growth_rate/100) ** (-(i * 5))
                population = prev_pop * survival_rates[i] * growth_effect
                
                if i >= 15:
                    population *= 0.6
            
            age_distribution.append(max(100, population))
        
        total_pop = sum(age_distribution)
        if total_pop > 0:
            age_percentages = [round(pop / total_pop * 100, 2) for pop in age_distribution]
        else:
            age_percentages = [5.5] * 3 + [4.0] * 10 + [2.0] * 4
        
        return age_percentages
    
    for yr in animation_years:
        year_data = pyramid_data[pyramid_data['year'] == yr]
        if year_data.empty:
            continue
        
        latest_data = year_data.iloc[0]
        tfr = latest_data.get('total_fertility_rate', 4.0)
        life_exp = latest_data.get('life_expectancy', 60)
        growth_rate = latest_data.get('population_growth_rate', 2.5)
        
        population_by_age = generate_realistic_age_distribution(tfr, life_exp, growth_rate)
        
        male_pop = [round(-pop * 0.515, 2) for pop in population_by_age]
        female_pop = [round(pop * 0.485, 2) for pop in population_by_age]
        
        fig.add_trace(go.Bar(
            y=age_groups, x=male_pop, name='Male',
            orientation='h', marker_color='lightblue',
            visible=(yr == animation_years[0])
        ))
        
        fig.add_trace(go.Bar(
            y=age_groups, x=female_pop, name='Female',
            orientation='h', marker_color='pink',
            visible=(yr == animation_years[0])
        ))
    
    if animate and len(animation_years) > 1:
        frames = []
        for yr in animation_years:
            year_data = pyramid_data[pyramid_data['year'] == yr]
            if year_data.empty:
                continue
            
            latest_data = year_data.iloc[0]
            tfr = latest_data.get('total_fertility_rate', 4.0)
            life_exp = latest_data.get('life_expectancy', 60)
            growth_rate = latest_data.get('population_growth_rate', 2.5)
            
            population_by_age = generate_realistic_age_distribution(tfr, life_exp, growth_rate)
            male_pop = [round(-pop * 0.515, 2) for pop in population_by_age]
            female_pop = [round(pop * 0.485, 2) for pop in population_by_age]
            
            frames.append(go.Frame(
                data=[
                    go.Bar(y=age_groups, x=male_pop, name='Male', marker_color='lightblue'),
                    go.Bar(y=age_groups, x=female_pop, name='Female', marker_color='pink')
                ],
                name=str(yr)
            ))
        
        fig.frames = frames
        
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {'label': 'Play', 'method': 'animate', 'args': [None]},
                    {'label': 'Pause', 'method': 'animate', 'args': [[None]]}
                ]
            }],
            sliders=[{
                'steps': [{'args': [[str(yr)]], 'label': str(yr), 'method': 'animate'} 
                         for yr in animation_years]
            }]
        )
    
    fig.update_layout(
        title=f"Population Pyramid - {country_name} ({year})",
        xaxis_title='Population (%)', yaxis_title='Age Groups',
        barmode='relative', height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_trend_comparison(df: pd.DataFrame, countries: list, indicators: list):
    """Comparaison tendances multi-pays avec valeurs arrondies"""
    
    if not countries or not indicators:
        st.warning("Please select countries and indicators for comparison")
        return
    
    trend_data = df[df['country_name'].isin(countries)].copy()
    
    fig = make_subplots(
        rows=len(indicators), cols=1,
        subplot_titles=[ind.replace('_', ' ').title() for ind in indicators],
        vertical_spacing=0.08
    )
    
    colors = px.colors.qualitative.Set1[:len(countries)]
    
    for i, indicator in enumerate(indicators):
        if indicator not in trend_data.columns:
            continue
            
        for j, country in enumerate(countries):
            country_data = trend_data[trend_data['country_name'] == country]
            clean_data = country_data[['year', indicator]].dropna()
            
            if not clean_data.empty:
                clean_data[indicator] = clean_data[indicator].round(2)
                
                fig.add_trace(
                    go.Scatter(
                        x=clean_data['year'], y=clean_data[indicator],
                        mode='lines+markers', name=country,
                        line=dict(color=colors[j]), showlegend=(i == 0)
                    ),
                    row=i+1, col=1
                )
    
    fig.update_layout(
        height=300 * len(indicators),
        title="Multi-Country Demographic Trends Comparison"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_clustering_visualization(clustered_data: pd.DataFrame):
    """Visualisation clustering avec valeurs arrondies"""
    
    if clustered_data.empty:
        return None, None
    
    available_indicators = ['total_fertility_rate', 'median_age', 'population_growth_rate', 'life_expectancy']
    valid_indicators = [ind for ind in available_indicators if ind in clustered_data.columns]
    
    if len(valid_indicators) < 2:
        return None, None
    
    x_indicator, y_indicator = valid_indicators[0], valid_indicators[1]
    
    clustered_data_display = clustered_data.copy()
    for indicator in [x_indicator, y_indicator]:
        clustered_data_display[indicator] = clustered_data_display[indicator].round(2)
    
    size_col = None
    if 'population_growth_rate' in clustered_data_display.columns:
        size_data = clustered_data_display['population_growth_rate'].fillna(0)
        size_col = np.abs(size_data) + 1
    
    fig_scatter = px.scatter(
        clustered_data_display, x=x_indicator, y=y_indicator,
        color='cluster_label' if 'cluster_label' in clustered_data_display.columns else 'cluster',
        size=size_col, hover_name='country_name',
        title='Country Clustering by Demographic Profile',
        labels={x_indicator: x_indicator.replace('_', ' ').title(),
                y_indicator: y_indicator.replace('_', ' ').title()}
    )
    
    fig_scatter.update_layout(height=500)
    
    cluster_summary = None
    if 'cluster_label' in clustered_data.columns:
        summary_indicators = [ind for ind in valid_indicators if ind in clustered_data.columns]
        
        cluster_stats = clustered_data.groupby('cluster_label').agg({
            **{ind: ['mean', 'count'] for ind in summary_indicators[:3]},
            'country_name': 'count'
        }).round(2)
        
        cluster_stats.columns = [f"{col[0]}_{col[1]}" if col[1] != 'count' or col[0] != 'country_name' 
                               else 'Countries' for col in cluster_stats.columns]
        
        cluster_summary = cluster_stats
    
    return fig_scatter, cluster_summary

def main():
    Config.setup_directories()

    if 'ml_config' not in st.session_state:
        st.session_state.ml_config = MultilingualConfig()
    
    ml_config = st.session_state.ml_config
    
    api_service = WorldBankAPIService()
    analytics = DemographicAnalytics()
    cache = CacheManager()
    debug = DebugTools()
    
    # Sidebar navigation
    st.sidebar.markdown("## 🌍 Language / Langue")
    language = st.sidebar.selectbox(
        "Select Language / Choisir la langue:",
        ["fr", "en"],
        index=0 if ml_config.get_language() == "fr" else 1,
        format_func=lambda x: "🇫🇷 Français" if x == "fr" else "🇬🇧 English"
    )
    
    if language != ml_config.get_language():
        ml_config.set_language(language)
        st.rerun()
    
    st.markdown(f'<h1 class="main-header">🌍 {ml_config.t("app_title")}</h1>', unsafe_allow_html=True)
    
    st.sidebar.markdown(f"## 🌍 {ml_config.t('navigation')}")
    
    menu_options = [
        ml_config.t("continental_overview"),
        ml_config.t("country_profiles"), 
        ml_config.t("trend_analysis"),
        ml_config.t("clustering_analysis"),
        ml_config.t("data_explorer"),
        ml_config.t("api_status"),
        ml_config.t("cache_management")
    ]
    
    page = st.sidebar.selectbox("Choose a view:", menu_options)
    
    # Data source info
    st.sidebar.markdown(f"## 📊 {ml_config.t('data_source')}")
    st.sidebar.info(f"""
    **{ml_config.t('world_bank_api')}**
    - {ml_config.t('real_time_data')}
    - {ml_config.t('african_countries')}
    - {ml_config.t('years_coverage')}
    - {ml_config.t('population_weighted')}
    """)
    
    # Data loading
    data_loading_text = "⚙️ Chargement des Données" if ml_config.get_language() == "fr" else "⚙️ Data Loading"
    st.sidebar.markdown(f"## {data_loading_text}")
    
    core_only_text = "Utiliser indicateurs principaux (plus rapide)" if ml_config.get_language() == "fr" else "Use core indicators only (faster)"
    use_core_indicators = st.sidebar.checkbox(core_only_text, value=False)
    
    reload_text = "🔄 Recharger Données" if ml_config.get_language() == "fr" else "🔄 Reload Data"
    if st.sidebar.button(reload_text):
        st.cache_data.clear()
        st.rerun()
    
    # Cache status
    st.sidebar.markdown(f"## 💾 {ml_config.t('cache_management')}")
    cache_info = cache.get_cache_info()
    
    if cache_info['total_files'] > 0:
        files_text = "fichiers en cache" if ml_config.get_language() == "fr" else "files cached"
        st.sidebar.success(f"🗃 {cache_info['total_files']} {files_text}")
        st.sidebar.caption(f"💾 {cache_info['total_size_mb']:.1f} MB")
        
        clear_text = "🗑️ Vider Cache" if ml_config.get_language() == "fr" else "🗑️ Clear Cache"
        if st.sidebar.button(clear_text):
            cache.clear_cache()
            st.rerun()
    else:
        no_cache_text = "🗃 Pas de fichiers cache" if ml_config.get_language() == "fr" else "🗃 No cache files"
        st.sidebar.info(no_cache_text)
    
    # Main content
    if page == ml_config.t("continental_overview"):
        try:
            with st.spinner(ml_config.t("loading_data")):
                df = load_demographic_data(use_core_only=use_core_indicators)
            
            if df.empty:
                st.error(ml_config.t("no_data"))
                return
            
            st.sidebar.success(f"✅ {df['country_iso2'].nunique()}/54 countries")
            
            create_multilingual_continental_overview(df, ml_config, analytics)
            
            if not df.empty:
                st.markdown("---")
                
                map_title = f"### 🗺️ {ml_config.t('select_indicator')}"
                st.markdown(map_title)
                
                available_indicators = [col for col in df.columns 
                                        if col in ['total_fertility_rate', 'population_growth_rate', 'median_age', 'dividend_score']]
                
                if available_indicators:
                    selected_indicator = st.selectbox(
                        f"{ml_config.t('select_indicator')}:",
                        available_indicators,
                        format_func=lambda x: ml_config.translator.get_indicator_name(x, ml_config.get_language())
                    )
                    
                    year_options = sorted(df['year'].unique(), reverse=True)
                    selected_year = st.selectbox(f"{ml_config.t('select_year')}:", year_options)
                    
                    create_multilingual_africa_map(df, selected_indicator, selected_year, ml_config)
                
                st.markdown("---")
                
                stats_title = f"### 📊 {ml_config.t('demographic_transition')}"
                st.markdown(stats_title)
                
                col_stats1, col_stats2 = st.columns(2)
                
                latest_year = df['year'].max()
                latest_data = df[df['year'] == latest_year]
                
                with col_stats1:
                    if 'total_fertility_rate' in df.columns:
                        lowest_fertility_text = "🏆 Taux de fécondité les plus bas:" if ml_config.get_language() == "fr" else "🏆 Lowest Fertility Rates:"
                        st.markdown(f"**{lowest_fertility_text}**")
                        top_tfr = latest_data.nsmallest(5, 'total_fertility_rate')[['country_name', 'total_fertility_rate']]
                        if not top_tfr.empty:
                            for _, row in top_tfr.iterrows():
                                tfr_formatted = ml_config.format_number(row['total_fertility_rate'], 1)
                                st.write(f"• {row['country_name']}: {tfr_formatted}")
                
                with col_stats2:
                    if 'population_growth_rate' in df.columns:
                        highest_growth_text = "📈 Taux de croissance les plus élevés:" if ml_config.get_language() == "fr" else "📈 Highest Growth Rates:"
                        st.markdown(f"**{highest_growth_text}**")
                        top_growth = latest_data.nlargest(5, 'population_growth_rate')[['country_name', 'population_growth_rate']]
                        if not top_growth.empty:
                            for _, row in top_growth.iterrows():
                                growth_formatted = ml_config.format_number(row['population_growth_rate'], 1)
                                st.write(f"• {row['country_name']}: {growth_formatted}%")
        
        except Exception as e:
            st.error(f"Error loading overview: {e}")
            debug_text = "Essayez la page Statut API et Debug" if ml_config.get_language() == "fr" else "Try the API Status & Debug page for diagnostics"
            st.info(debug_text)
    
    elif page == ml_config.t("country_profiles"):
        try:
            df = load_demographic_data(use_core_only=use_core_indicators)
            
            if df.empty or 'country_name' not in df.columns:
                st.error(ml_config.t("no_data"))
                return
            
            profiles_header = f'<div class="section-header">🏛️ {ml_config.t("country_profiles")} - {ml_config.t("demographic_dividend_desc")}</div>'
            st.markdown(profiles_header, unsafe_allow_html=True)
            
            available_countries = sorted(df['country_name'].unique())
            select_text = f"{ml_config.t('select_country')} profil détaillé:" if ml_config.get_language() == "fr" else "Select country for detailed profile:"
            selected_country = st.selectbox(select_text, available_countries)
            
            country_data = df[df['country_name'] == selected_country].copy()
            
            if not country_data.empty:
                latest_data = country_data[country_data['year'] == country_data['year'].max()].iloc[0]
                
                overview_text = "Aperçu du Profil" if ml_config.get_language() == "fr" else "Profile Overview"
                st.markdown(f"### 📊 {selected_country} - {overview_text} ({latest_data['year']:.0f})")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'total_fertility_rate' in latest_data.index and pd.notna(latest_data['total_fertility_rate']):
                        tfr_val = ml_config.format_number(latest_data['total_fertility_rate'], 1)
                        st.metric(ml_config.t("fertility_rate"), tfr_val)
                    else:
                        st.metric(ml_config.t("fertility_rate"), "N/A")
                
                with col2:
                    if 'total_population' in latest_data.index and pd.notna(latest_data['total_population']):
                        pop_millions = latest_data['total_population'] / 1e6
                        pop_val = f"{ml_config.format_number(pop_millions, 1)}M"
                        st.metric(ml_config.t("population"), pop_val)
                    else:
                        st.metric(ml_config.t("population"), "N/A")
                
                with col3:
                    if 'median_age' in latest_data.index and pd.notna(latest_data['median_age']):
                        age_val = f"{ml_config.format_number(latest_data['median_age'], 1)} {ml_config.t('years')}"
                        st.metric(ml_config.t("median_age"), age_val)
                    else:
                        st.metric(ml_config.t("median_age"), "N/A")
                
                with col4:
                    if 'dividend_status' in latest_data.index:
                        status = latest_data['dividend_status']
                        color = {'High Opportunity': '🟢', 'Opening Window': '🟡', 'Limited Window': '🔴', 'No Window': '⚪'}.get(status, '❓')
                        status_translated = ml_config.translator.get_text(status.lower().replace(" ", "_"), ml_config.get_language())
                        st.metric(ml_config.t("demographic_dividend"), f"{color} {status_translated}")
                    else:
                        st.metric(ml_config.t("demographic_dividend"), "N/A")
                
                pyramid_title = f"### 📈 {ml_config.t('population_pyramid')} - Analyse"
                st.markdown(pyramid_title)
                
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    select_year_text = f"{ml_config.t('select_year')}:"
                    pyramid_year = st.selectbox(select_year_text, sorted(country_data['year'].unique(), reverse=True))
                    
                    animate_text = "🎬 Animer dans le temps (1990-2023)" if ml_config.get_language() == "fr" else "🎬 Animate over time (1990-2023)"
                    animate_pyramid = st.checkbox(animate_text)
                    
                    if animate_pyramid:
                        info_text = "🎬 L'animation montrera la transition démographique" if ml_config.get_language() == "fr" else "🎬 Animation will show demographic transition over time"
                        st.info(info_text)
                
                with col1:
                    create_population_pyramid(df, selected_country, pyramid_year, animate_pyramid)
                
                trends_title = f"### 📈 {ml_config.t('trend_analysis')} - Historiques"
                st.markdown(trends_title)
                
                numeric_cols = country_data.select_dtypes(include=['number']).columns
                available_for_trend = [col for col in numeric_cols 
                                     if col not in ['year'] and country_data[col].notna().sum() > 3]
                
                if available_for_trend:
                    indicators_text = f"{ml_config.t('select_indicator')} à afficher:" if ml_config.get_language() == "fr" else "Select indicators to display:"
                    selected_indicators = st.multiselect(
                        indicators_text,
                        available_for_trend,
                        default=available_for_trend[:3],
                        format_func=lambda x: ml_config.translator.get_indicator_name(x, ml_config.get_language())
                    )
                    
                    if selected_indicators:
                        create_trend_comparison(country_data, [selected_country], selected_indicators)
        
        except Exception as e:
            st.error(f"Error in country profiles: {e}")
    
    elif page == ml_config.t("trend_analysis"):
        try:
            df = load_demographic_data(use_core_only=use_core_indicators)
            
            if df.empty:
                st.error(ml_config.t("no_data"))
                return
            
            trends_header = f'<div class="section-header">📈 {ml_config.t("trend_analysis")} - {ml_config.t("countries_comparison")}</div>'
            st.markdown(trends_header, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                countries_title = f"**{ml_config.t('select_country')} pour comparaison:**" if ml_config.get_language() == "fr" else "**Select Countries for Comparison:**"
                st.markdown(countries_title)
                available_countries = sorted(df['country_name'].unique())
                comparison_text = f"{ml_config.t('countries_comparison')} (sélectionner 2-6 pour visualisation optimale):" if ml_config.get_language() == "fr" else "Countries (select 2-6 for best visualization):"
                selected_countries = st.multiselect(
                    comparison_text,
                    available_countries,
                    default=available_countries[:4] if len(available_countries) >= 4 else available_countries,
                    max_selections=6
                )
            
            with col2:
                indicators_title = f"**{ml_config.t('select_indicator')}:**"
                st.markdown(indicators_title)
                numeric_cols = df.select_dtypes(include=['number']).columns
                available_indicators = [col for col in numeric_cols 
                                      if col not in ['year'] and df[col].notna().sum() > 50]
                
                demographic_text = f"Indicateurs démographiques:" if ml_config.get_language() == "fr" else "Demographic indicators:"
                selected_indicators = st.multiselect(
                    demographic_text,
                    available_indicators,
                    default=['total_fertility_rate', 'population_growth_rate'] if all(x in available_indicators for x in ['total_fertility_rate', 'population_growth_rate']) else available_indicators[:2],
                    format_func=lambda x: ml_config.translator.get_indicator_name(x, ml_config.get_language())
                )
            
            if selected_countries and selected_indicators:
                create_trend_comparison(df, selected_countries, selected_indicators)
                
                stats_title = f"### 📊 {ml_config.t('demographic_transition')} Statistiques"
                st.markdown(stats_title)
                
                comparison_analysis = analytics.generate_country_comparison(df, selected_countries, selected_indicators)
                
                if comparison_analysis:
                    latest_title = "#### Comparaison Valeurs Récentes" if ml_config.get_language() == "fr" else "#### Latest Values Comparison"
                    st.markdown(latest_title)
                    for indicator, data in comparison_analysis['latest_comparison'].items():
                        indicator_name = ml_config.translator.get_indicator_name(indicator, ml_config.get_language())
                        st.markdown(f"**{indicator_name}:**")
                        ranking_df = pd.DataFrame(data['ranking'], columns=['Pays' if ml_config.get_language() == "fr" else 'Country', 'Valeur' if ml_config.get_language() == "fr" else 'Value'])
                        st.dataframe(ranking_df, hide_index=True)
                    
                    if comparison_analysis['correlations']:
                        corr_title = "#### Corrélations entre Indicateurs" if ml_config.get_language() == "fr" else "#### Indicator Correlations"
                        st.markdown(corr_title)
                        corr_data = []
                        for pair, corr in comparison_analysis['correlations'].items():
                            indicators_pair = pair.replace('_vs_', ' vs ').replace('_', ' ').title()
                            corr_label = "Indicateurs" if ml_config.get_language() == "fr" else "Indicators"
                            corr_value = "Corrélation" if ml_config.get_language() == "fr" else "Correlation"
                            corr_data.append({corr_label: indicators_pair, corr_value: f"{corr:.3f}"})
                        
                        if corr_data:
                            st.dataframe(pd.DataFrame(corr_data), hide_index=True)
            else:
                select_text = f"Veuillez sélectionner des {ml_config.t('countries_comparison').lower()} et {ml_config.t('select_indicator').lower()}" if ml_config.get_language() == "fr" else "Please select countries and indicators to begin analysis"
                st.info(select_text)
        
        except Exception as e:
            st.error(f"Error in trend analysis: {e}")
    
    elif page == ml_config.t("clustering_analysis"):
        try:
            df = load_demographic_data(use_core_only=use_core_indicators)
            
            if df.empty:
                st.error(ml_config.t("no_data"))
                return
            
            clustering_header = f'<div class="section-header">🔬 {ml_config.t("clustering_analysis")} - {ml_config.t("ml_clustering_desc")}</div>'
            st.markdown(clustering_header, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                year_text = f"{ml_config.t('select_year')} pour analyse clustering:"
                cluster_year = st.selectbox(year_text, sorted(df['year'].unique(), reverse=True))
            
            with col2:
                method_text = f"Méthode K-Means optimisée" if ml_config.get_language() == "fr" else f"Optimized K-Means Method"
                indicators_used = f"Indicateurs Utilisés" if ml_config.get_language() == "fr" else "Indicators Used"
                classification = f"Classification: Stades de {ml_config.t('demographic_transition').lower()}" if ml_config.get_language() == "fr" else f"Classification: {ml_config.t('demographic_transition')} stages"
                
                st.info(f"""
                **{method_text}**
                **{indicators_used}:** {', '.join([ml_config.translator.get_indicator_name(ind, ml_config.get_language()) for ind in Config.CLUSTERING_CONFIG['indicators']])}
                **{classification}**
                """)
            
            clustered_data = analytics.get_country_clusters(df, cluster_year)
            
            if not clustered_data.empty:
                clusters_title = f"### 🎯 {ml_config.t('countries_comparison')} par {ml_config.t('demographic_transition')}"
                st.markdown(clusters_title)
                
                fig_scatter, cluster_summary = create_clustering_visualization(clustered_data)
                
                if fig_scatter:
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                if cluster_summary is not None:
                    characteristics_title = f"### 📊 Caractéristiques des Clusters" if ml_config.get_language() == "fr" else "### 📊 Cluster Characteristics"
                    st.markdown(characteristics_title)
                    st.dataframe(cluster_summary)
                
                countries_by_stage = f"### 🗂️ {ml_config.t('countries_comparison')} par Stade de {ml_config.t('demographic_transition')}"
                st.markdown(countries_by_stage)
                
                if 'cluster_label' in clustered_data.columns:
                    for cluster_label in sorted(clustered_data['cluster_label'].unique()):
                        cluster_countries = clustered_data[clustered_data['cluster_label'] == cluster_label]['country_name'].tolist()
                        
                        if 'High Fertility' in cluster_label:
                            color = "🔴"
                        elif 'Moderate' in cluster_label:
                            color = "🟡"
                        elif 'Advanced' in cluster_label:
                            color = "🟢"
                        else:
                            color = "🔵"
                        
                        countries_text = "pays" if ml_config.get_language() == "fr" else "countries"
                        st.markdown(f"**{color} {cluster_label}** ({len(cluster_countries)} {countries_text}):")
                        st.write(", ".join(cluster_countries))
            else:
                insufficient_text = f"Données insuffisantes pour l'analyse clustering. Essayez une autre année." if ml_config.get_language() == "fr" else "Insufficient data for clustering analysis. Try a different year or check data availability."
                st.warning(insufficient_text)
        
        except Exception as e:
            st.error(f"Error in clustering analysis: {e}")
    
    elif page == ml_config.t("data_explorer"):
        try:
            df = load_demographic_data(use_core_only=use_core_indicators)
            
            if df.empty:
                st.error(ml_config.t("no_data"))
                return
            
            explorer_header = f'<div class="section-header">🔍 {ml_config.t("data_explorer")} - {ml_config.t("advanced_filtering")}</div>'
            st.markdown(explorer_header, unsafe_allow_html=True)
            
            filters_title = f"### 🎛️ {ml_config.t('advanced_filtering')}"
            st.markdown(filters_title)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                countries_title = f"**{ml_config.t('countries_comparison')}:**"
                st.markdown(countries_title)
                available_countries = sorted(df['country_name'].unique())
                select_countries_text = f"{ml_config.t('select_country')}:"
                selected_countries_explorer = st.multiselect(
                    select_countries_text,
                    available_countries,
                    default=available_countries,
                    key="explorer_countries"
                )
            
            with col2:
                years_title = f"**{ml_config.t('years')}:**"
                st.markdown(years_title)
                year_range_text = f"Plage d'{ml_config.t('years').lower()}:" if ml_config.get_language() == "fr" else "Year range:"
                year_range = st.slider(
                    year_range_text,
                    min_value=int(df['year'].min()),
                    max_value=int(df['year'].max()),
                    value=(int(df['year'].min()), int(df['year'].max())),
                    key="explorer_years"
                )
            
            with col3:
                indicators_title = f"**{ml_config.t('select_indicator')}:**"
                st.markdown(indicators_title)
                numeric_cols = df.select_dtypes(include=['number']).columns
                available_indicators_explorer = [col for col in numeric_cols if col not in ['year']]
                indicators_select_text = f"{ml_config.t('select_indicator')}:"
                selected_indicators_explorer = st.multiselect(
                    indicators_select_text,
                    available_indicators_explorer,
                    default=available_indicators_explorer[:5],
                    format_func=lambda x: ml_config.translator.get_indicator_name(x, ml_config.get_language()),
                    key="explorer_indicators"
                )
            
            filtered_df = df[
                (df['country_name'].isin(selected_countries_explorer)) &
                (df['year'] >= year_range[0]) &
                (df['year'] <= year_range[1])
            ].copy()
            
            display_cols = ['country_name', 'year'] + selected_indicators_explorer
            display_df = filtered_df[display_cols].copy()
            
            for col in selected_indicators_explorer:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(2)
            
            summary_title = f"### 📊 {ml_config.t('data_source')} Filtrées"
            st.markdown(summary_title)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                countries_label = ml_config.t('countries_comparison') if ml_config.get_language() == "fr" else "Countries"
                st.metric(countries_label, len(selected_countries_explorer))
            with col2:
                years_label = ml_config.t('years')
                st.metric(years_label, f"{year_range[1] - year_range[0] + 1}")
            with col3:
                indicators_label = ml_config.t('select_indicator') if ml_config.get_language() == "fr" else "Indicators"
                st.metric(indicators_label, len(selected_indicators_explorer))
            with col4:
                records_label = "Enregistrements" if ml_config.get_language() == "fr" else "Records"
                st.metric(records_label, len(display_df))
            
            dataset_title = f"### 📋 Jeu de {ml_config.t('data_source')} Filtré"
            st.markdown(dataset_title)
            st.dataframe(display_df, use_container_width=True)
            
            if selected_indicators_explorer:
                stats_title = f"### 📈 Statistiques Descriptives" if ml_config.get_language() == "fr" else "### 📈 Descriptive Statistics"
                st.markdown(stats_title)
                
                stats_df = display_df[selected_indicators_explorer].describe().round(2)
                st.dataframe(stats_df)
            
            export_title = f"### {ml_config.t('export_options')}"
            st.markdown(export_title)
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = display_df.to_csv(index=False)
                filename_csv = f"afrique_demographique_filtre_{datetime.now().strftime('%Y%m%d')}.csv"
                if ml_config.get_language() == "en":
                    filename_csv = f"africa_demographics_filtered_{datetime.now().strftime('%Y%m%d')}.csv"
                
                st.download_button(
                    ml_config.t("download_csv"),
                    csv_data,
                    filename_csv,
                    "text/csv",
                    key="download_csv"
                )
            
            with col2:
                json_data = display_df.to_json(orient='records', indent=2)
                filename_json = f"afrique_demographique_filtre_{datetime.now().strftime('%Y%m%d')}.json"
                if ml_config.get_language() == "en":
                    filename_json = f"africa_demographics_filtered_{datetime.now().strftime('%Y%m%d')}.json"
                
                st.download_button(
                    ml_config.t("download_json"),
                    json_data,
                    filename_json,
                    "application/json",
                    key="download_json"
                )
        
        except Exception as e:
            st.error(f"Error in data explorer: {e}")
    
    elif page == ml_config.t("api_status"):
        status_header = f'<div class="section-header">🔌 {ml_config.t("api_status")}</div>'
        st.markdown(status_header, unsafe_allow_html=True)
        
        connectivity_text = "🌐 Tester la connectivité de base" if ml_config.get_language() == "fr" else "🌐 Test Basic Connectivity"
        if st.button(connectivity_text):
            testing_text = "Test en cours..." if ml_config.get_language() == "fr" else "Testing connectivity..."
            with st.spinner(testing_text):
                results = debug.test_basic_connectivity()
                
                for test in results['tests']:
                    if test['status'] in ['RÉUSSI', 'PASS']:
                        st.success(f"✅ {test['test']}: {test['details']}")
                    else:
                        st.error(f"❌ {test['test']}: {test['details']}")
        
        st.markdown("---")
        
        comprehensive_text = "🧪 Exécuter test système complet" if ml_config.get_language() == "fr" else "🧪 Run Comprehensive Test"
        if st.button(comprehensive_text):
            debug.run_comprehensive_test()
        
        st.markdown("---")
        
        individual_test_title = f"### 🔍 Test d'Indicateurs Individuels" if ml_config.get_language() == "fr" else "### 🔍 Test Individual Indicators"
        st.markdown(individual_test_title)
        
        indicator_text = f"{ml_config.t('select_indicator')} à tester:" if ml_config.get_language() == "fr" else "Select indicator to test:"
        test_indicator = st.selectbox(
            indicator_text,
            list(Config.CORE_INDICATORS.keys()),
            format_func=lambda x: ml_config.translator.get_indicator_name(Config.CORE_INDICATORS[x], ml_config.get_language())
        )
        
        test_button_text = "Tester l'Indicateur Sélectionné" if ml_config.get_language() == "fr" else "Test Selected Indicator"
        if st.button(test_button_text):
            testing_indicator_text = "Test de l'indicateur..." if ml_config.get_language() == "fr" else "Testing indicator..."
            with st.spinner(testing_indicator_text):
                result = debug.test_single_indicator(test_indicator)
                
                if result.get('status') in ['SUCCÈS', 'SUCCESS']:
                    success_text = f"Succès: {result.get('data_count', 0)} enregistrements" if ml_config.get_language() == "fr" else f"Success: {result.get('data_count', 0)} records"
                    st.success(f"✅ {success_text}")
                    if result.get('sample_record'):
                        st.json(result['sample_record'])
                else:
                    error_text = f"Échec: {result.get('error', 'Erreur inconnue')}" if ml_config.get_language() == "fr" else f"Failed: {result.get('error', 'Unknown error')}"
                    st.error(f"❌ {error_text}")
    
    elif page == ml_config.t("cache_management"):
        cache_header = f'<div class="section-header">💾 {ml_config.t("cache_management")}</div>'
        st.markdown(cache_header, unsafe_allow_html=True)
        
        cache_info = cache.get_cache_info()
        
        if cache_info['total_files'] == 0:
            no_files_text = "Aucun fichier en cache trouvé" if ml_config.get_language() == "fr" else "No cached files found"
            st.info(no_files_text)
        else:
            files_text = "fichiers en cache trouvés" if ml_config.get_language() == "fr" else "cached files found"
            st.success(f"{cache_info['total_files']} {files_text} ({cache_info['total_size_mb']:.1f} MB)")
            
            if cache_info['files']:
                cache_df = pd.DataFrame(cache_info['files'])
                cache_df['age_hours'] = cache_df['age_hours'].round(1)
                cache_df['size_kb'] = cache_df['size_kb'].round(1)
                
                if ml_config.get_language() == "fr":
                    cache_df = cache_df.rename(columns={
                        'name': 'Nom',
                        'size_kb': 'Taille (KB)',
                        'age_hours': 'Âge (heures)',
                        'valid': 'Valide'
                    })
                
                st.dataframe(cache_df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            clear_all_text = "🗑️ Vider Tout le Cache" if ml_config.get_language() == "fr" else "🗑️ Clear All Cache"
            if st.button(clear_all_text):
                cache.clear_cache()
                st.rerun()
        
        with col2:
            refresh_text = "🔄 Actualiser les Infos" if ml_config.get_language() == "fr" else "🔄 Refresh Info"
            if st.button(refresh_text):
                st.rerun()
        
        with col3:
            expire_text = f"Le cache expire après {Config.CACHE_HOURS} heures" if ml_config.get_language() == "fr" else f"Cache expires after {Config.CACHE_HOURS} hours"
            st.info(expire_text)
    
    # Footer multilingue
    st.markdown("---")
    current_date = datetime.now()
    if ml_config.get_language() == "fr":
        date_str = current_date.strftime("%d %B %Y")
    else:
        date_str = current_date.strftime("%B %d, %Y")
    
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        <p>🌍 {ml_config.t("platform_description")} - Conception et développement Zakaria Benhoumad</p>
        <p>📊 {ml_config.t("features_list")}</p>
        <p>🔗 {ml_config.t("data_attribution")} • {date_str}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()