# ==================================================
# File: multilingual_config.py
# Configuration multilingue + Config original
# ==================================================

from pathlib import Path
from translations import TranslationManager

# Configuration originale pour compatibilité avec les imports existants
class Config:
    """Configuration originale pour compatibilité avec les imports existants"""
    
    # API Configuration
    WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2"
    API_TIMEOUT = 60
    REQUEST_DELAY = 0.3
    
    # Cache Configuration
    CACHE_DIR = Path("data_cache")
    CACHE_HOURS = 24
    
    # African countries (ISO2 codes as returned by World Bank API)
    AFRICAN_COUNTRIES = {
        'DZ': 'Algeria', 'AO': 'Angola', 'BJ': 'Benin', 'BW': 'Botswana',
        'BF': 'Burkina Faso', 'BI': 'Burundi', 'CM': 'Cameroon', 'CV': 'Cabo Verde',
        'CF': 'Central African Republic', 'TD': 'Chad', 'KM': 'Comoros', 'CG': 'Congo, Rep.',
        'CD': 'Congo, Dem. Rep.', 'CI': "Cote d'Ivoire", 'DJ': 'Djibouti',
        'EG': 'Egypt, Arab Rep.', 'GQ': 'Equatorial Guinea', 'ER': 'Eritrea', 'SZ': 'Eswatini',
        'ET': 'Ethiopia', 'GA': 'Gabon', 'GM': 'Gambia, The', 'GH': 'Ghana',
        'GN': 'Guinea', 'GW': 'Guinea-Bissau', 'KE': 'Kenya', 'LS': 'Lesotho',
        'LR': 'Liberia', 'LY': 'Libya', 'MG': 'Madagascar', 'MW': 'Malawi',
        'ML': 'Mali', 'MR': 'Mauritania', 'MU': 'Mauritius', 'MA': 'Morocco',
        'MZ': 'Mozambique', 'NA': 'Namibia', 'NE': 'Niger', 'NG': 'Nigeria',
        'RW': 'Rwanda', 'ST': 'Sao Tome and Principe', 'SN': 'Senegal',
        'SC': 'Seychelles', 'SL': 'Sierra Leone', 'SO': 'Somalia', 'ZA': 'South Africa',
        'SS': 'South Sudan', 'SD': 'Sudan', 'TZ': 'Tanzania', 'TG': 'Togo',
        'TN': 'Tunisia', 'UG': 'Uganda', 'ZM': 'Zambia', 'ZW': 'Zimbabwe'
    }
    
    # World Bank indicators
    INDICATORS = {
        'SP.DYN.TFRT.IN': 'total_fertility_rate',
        'SP.POP.GROW': 'population_growth_rate',
        'SP.POP.0014.TO.ZS': 'population_0_14_percent',
        'SP.POP.1564.TO.ZS': 'population_15_64_percent',
        'SP.POP.65UP.TO.ZS': 'population_65_plus_percent',
        'SP.DYN.LE00.IN': 'life_expectancy',
        'SP.POP.TOTL': 'total_population',
        'SP.DYN.CBRT.IN': 'birth_rate',
        'SP.DYN.CDRT.IN': 'death_rate',
        'SP.URB.TOTL.IN.ZS': 'urban_population_percent'
    }
    
    # Core indicators for testing
    CORE_INDICATORS = {
        'SP.DYN.TFRT.IN': 'total_fertility_rate',
        'SP.POP.TOTL': 'total_population',
        'SP.POP.GROW': 'population_growth_rate',
        'SP.POP.0014.TO.ZS': 'population_0_14_percent'
    }
    
    # Demographic dividend thresholds
    DIVIDEND_THRESHOLDS = {
        'high_opportunity': {'child_dependency': 45, 'old_dependency': 15, 'working_age': 60},
        'opening_window': {'child_dependency': 55, 'old_dependency': 20, 'working_age': 55},
        'limited_window': {'child_dependency': 65, 'old_dependency': 25, 'working_age': 50}
    }
    
    # Clustering configuration
    CLUSTERING_CONFIG = {
        'n_clusters': 4,
        'random_state': 42,
        'indicators': ['total_fertility_rate', 'median_age', 'population_growth_rate', 'life_expectancy'],
        'cluster_labels': [
            'Early Transition (High Fertility)',
            'Moderate Transition', 
            'Advanced Transition',
            'Late Transition (Low Fertility)'
        ]
    }
    
    # Visualization settings
    VIZ_CONFIG = {
        'color_schemes': {
            'fertility': 'RdYlBu_r',
            'age': 'Viridis',
            'growth': 'Plasma',
            'dividend': 'RdYlGn'
        },
        'default_years': list(range(1990, 2024)),
        'animation_duration': 800
    }
    
    @classmethod
    def setup_directories(cls):
        """Create necessary project directories"""
        cls.CACHE_DIR.mkdir(exist_ok=True)

class MultilingualConfig:
    """Configuration multilingue de la plateforme"""
    
    def __init__(self):
        self.current_language = "fr"  # Langue par défaut
        self.translator = TranslationManager()
    
    def set_language(self, language: str):
        """Définir la langue active"""
        if language in ["fr", "en"]:
            self.current_language = language
    
    def get_language(self) -> str:
        """Obtenir la langue active"""
        return self.current_language
    
    def t(self, key: str) -> str:
        """Raccourci pour traduction"""
        return self.translator.get_text(key, self.current_language)
    
    def format_number(self, number: float, decimals: int = 1) -> str:
        """Formatage des nombres selon la locale"""
        if self.current_language == "fr":
            # Format français : virgule décimale, espace milliers
            formatted = f"{number:,.{decimals}f}".replace(",", " ").replace(".", ",")
            return formatted.replace(" ", "\u00a0")  # Espace insécable
        else:
            # Format anglais : point décimal, virgule milliers
            return f"{number:,.{decimals}f}"
    
    def get_date_format(self) -> str:
        """Format de date selon la langue"""
        if self.current_language == "fr":
            return "%d/%m/%Y"
        else:
            return "%m/%d/%Y"