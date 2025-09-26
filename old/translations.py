# ==================================================
# File: translations.py
# Multilingual Support for Africa Demographics Platform
# ==================================================

import streamlit as st

class TranslationManager:
    """Gestionnaire de traductions avec terminologie scientifique"""
    
    TRANSLATIONS = {
        # Navigation et titres principaux
        "app_title": {
            "fr": "Plateforme Démographique Africaine",
            "en": "Africa Demographics Platform"
        },
        "navigation": {
            "fr": "Navigation",
            "en": "Navigation"
        },
        "continental_overview": {
            "fr": "Vue Continentale",
            "en": "Continental Overview"
        },
        "country_profiles": {
            "fr": "Profils Pays",
            "en": "Country Profiles"
        },
        "trend_analysis": {
            "fr": "Analyse des Tendances",
            "en": "Trend Analysis"
        },
        "clustering_analysis": {
            "fr": "Analyse par Groupement",
            "en": "Clustering Analysis"
        },
        "data_explorer": {
            "fr": "Explorateur de Données",
            "en": "Data Explorer"
        },
        "api_status": {
            "fr": "Statut API et Diagnostic",
            "en": "API Status & Debug"
        },
        "cache_management": {
            "fr": "Gestion du Cache",
            "en": "Cache Management"
        },
        
        # Terminologie démographique scientifique
        "total_fertility_rate": {
            "fr": "Indice Synthétique de Fécondité",
            "en": "Total Fertility Rate"
        },
        "population_growth_rate": {
            "fr": "Taux d'Accroissement Démographique",
            "en": "Population Growth Rate"
        },
        "median_age": {
            "fr": "Âge Médian",
            "en": "Median Age"
        },
        "life_expectancy": {
            "fr": "Espérance de Vie à la Naissance",
            "en": "Life Expectancy at Birth"
        },
        "dependency_ratio": {
            "fr": "Rapport de Dépendance",
            "en": "Dependency Ratio"
        },
        "child_dependency_ratio": {
            "fr": "Rapport de Dépendance Juvénile",
            "en": "Child Dependency Ratio"
        },
        "old_age_dependency_ratio": {
            "fr": "Rapport de Dépendance des Personnes Âgées",
            "en": "Old-Age Dependency Ratio"
        },
        "demographic_dividend": {
            "fr": "Dividende Démographique",
            "en": "Demographic Dividend"
        },
        "demographic_transition": {
            "fr": "Transition Démographique",
            "en": "Demographic Transition"
        },
        "population_pyramid": {
            "fr": "Pyramide des Âges",
            "en": "Population Pyramid"
        },
        "age_structure": {
            "fr": "Structure par Âge",
            "en": "Age Structure"
        },
        "working_age_population": {
            "fr": "Population en Âge de Travailler",
            "en": "Working-Age Population"
        },
        
        # Statuts du dividende démographique
        "high_opportunity": {
            "fr": "Opportunité Maximale",
            "en": "High Opportunity"
        },
        "opening_window": {
            "fr": "Fenêtre qui s'Ouvre",
            "en": "Opening Window"
        },
        "limited_window": {
            "fr": "Fenêtre Limitée",
            "en": "Limited Window"
        },
        "no_window": {
            "fr": "Aucune Fenêtre",
            "en": "No Window"
        },
        
        # Classifications de clustering
        "early_transition": {
            "fr": "Transition Précoce (Haute Fécondité)",
            "en": "Early Transition (High Fertility)"
        },
        "moderate_transition": {
            "fr": "Transition Modérée",
            "en": "Moderate Transition"
        },
        "advanced_transition": {
            "fr": "Transition Avancée",
            "en": "Advanced Transition"
        },
        "late_transition": {
            "fr": "Transition Tardive (Faible Fécondité)",
            "en": "Late Transition (Low Fertility)"
        },
        
        # Interface utilisateur
        "select_country": {
            "fr": "Sélectionner un pays",
            "en": "Select country"
        },
        "select_year": {
            "fr": "Sélectionner l'année",
            "en": "Select year"
        },
        "select_indicator": {
            "fr": "Sélectionner l'indicateur",
            "en": "Select indicator"
        },
        "countries_comparison": {
            "fr": "Comparaison de pays",
            "en": "Countries comparison"
        },
        "data_source": {
            "fr": "Source des Données",
            "en": "Data Source"
        },
        "world_bank_api": {
            "fr": "API de la Banque Mondiale",
            "en": "World Bank Open Data API"
        },
        "real_time_data": {
            "fr": "Données en temps réel",
            "en": "Real-time data"
        },
        "african_countries": {
            "fr": "54 pays africains",
            "en": "54 African countries"
        },
        "years_coverage": {
            "fr": "Années : 1990-2023",
            "en": "Years: 1990-2023"
        },
        "population_weighted": {
            "fr": "Métriques pondérées par population",
            "en": "Population-weighted metrics"
        },
        
        # Métriques et statistiques
        "population": {
            "fr": "Population",
            "en": "Population"
        },
        "fertility_rate": {
            "fr": "Taux de Fécondité",
            "en": "Fertility Rate"
        },
        "growth_rate": {
            "fr": "Taux de Croissance",
            "en": "Growth Rate"
        },
        "children_per_woman": {
            "fr": "enfants/femme",
            "en": "children/woman"
        },
        "percent_per_year": {
            "fr": "%/an",
            "en": "%/year"
        },
        "years": {
            "fr": "ans",
            "en": "years"
        },
        "million": {
            "fr": "millions",
            "en": "million"
        },
        "billion": {
            "fr": "milliards",
            "en": "billion"
        },
        
        # Messages système
        "loading_data": {
            "fr": "Chargement des données démographiques...",
            "en": "Loading demographic data..."
        },
        "no_data": {
            "fr": "Aucune donnée disponible",
            "en": "No data available"
        },
        "data_loaded": {
            "fr": "Données chargées avec succès",
            "en": "Data loaded successfully"
        },
        "api_connected": {
            "fr": "API connectée avec succès",
            "en": "API connected successfully"
        },
        "api_error": {
            "fr": "Erreur de connexion API",
            "en": "API connection error"
        },
        "cache_cleared": {
            "fr": "Cache vidé avec succès",
            "en": "Cache cleared successfully"
        },
        
        # Descriptions et aide
        "population_calculation": {
            "fr": "Calculé à partir des données pays de la Banque Mondiale",
            "en": "Calculated from World Bank country-level data"
        },
        "demographic_dividend_desc": {
            "fr": "Suivi en temps réel des opportunités économiques",
            "en": "Real-time tracking of economic opportunities"
        },
        "ml_clustering_desc": {
            "fr": "Regroupement ML par profils démographiques",
            "en": "ML clustering by demographic profiles"
        },
        "advanced_filtering": {
            "fr": "Filtrage avancé et export personnalisé",
            "en": "Advanced filtering and custom export"
        },
        
        # Export et téléchargement
        "download_csv": {
            "fr": "Télécharger en CSV",
            "en": "Download as CSV"
        },
        "download_json": {
            "fr": "Télécharger en JSON",
            "en": "Download as JSON"
        },
        "export_options": {
            "fr": "Options d'Export",
            "en": "Export Options"
        },
        
        # Footer
        "platform_description": {
            "fr": "Plateforme Démographique Africaine - Implémentation Complète",
            "en": "Africa Demographics Platform - Complete Implementation"
        },
        "features_list": {
            "fr": "Fonctionnalités : Vue Continentale • Profils Pays • Analyse Tendances • Groupement ML • Explorateur",
            "en": "Features: Continental Overview • Country Profiles • Trend Analysis • ML Clustering • Data Explorer"
        },
        "data_attribution": {
            "fr": "API Banque Mondiale • Données Temps Réel",
            "en": "World Bank API • Real-time Data"
        }
    }
    
    @classmethod
    def get_text(cls, key: str, lang: str = "fr") -> str:
        """Récupérer texte traduit avec fallback"""
        try:
            return cls.TRANSLATIONS[key][lang]
        except KeyError:
            # Fallback vers français si clé inexistante
            return cls.TRANSLATIONS.get(key, {}).get("fr", key)
    
    @classmethod
    def get_indicator_name(cls, technical_name: str, lang: str = "fr") -> str:
        """Convertir nom technique en nom scientifique traduit"""
        return cls.get_text(technical_name, lang)