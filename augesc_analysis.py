# ============================================
# ANALYSE DES DONNÉES CONVERSATIONNELLES AUGESC
# Dashboard complet avec tous les modèles ML et visualisations
# ============================================

# ---------- 1. IMPORTATION DES LIBRAIRIES ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Modèles ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Métriques
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve, 
    f1_score, precision_score, recall_score
)

# Statistiques
import scipy.stats as stats
from scipy.stats import shapiro, levene, mannwhitneyu

# Utilitaires
import json
from datetime import datetime
import os
import pickle

# Configuration avancée des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# ---------- 2. CHARGEMENT ET INSPECTION ----------
print("="*80)
print("CHARGEMENT ET INSPECTION DES DONNÉES")
print("="*80)

# Chargement
try:
    df = pd.read_csv("augesc_export_ready.csv")
    print(f"📊 Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Afficher les premières lignes
    print("\n📋 Aperçu des données (5 premières lignes):")
    print(df.head())
    
except FileNotFoundError:
    print("❌ Fichier 'augesc_export_ready.csv' non trouvé!")
    print("⚠️  Assurez-vous que le fichier est dans le bon répertoire.")
    exit()

# Inspection détaillée
print("\n🔍 INSPECTION DES COLONNES:")
print("="*50)
for i, col in enumerate(df.columns, 1):
    dtype = str(df[col].dtype)
    unique = df[col].nunique()
    missing = df[col].isnull().sum()
    print(f"{i:2d}. {col:<25} {dtype:<15} {unique:>5} valeurs uniques, {missing:>5} manquants")
    
    # Afficher des exemples pour les colonnes avec peu de valeurs uniques
    if unique <= 5 and unique > 0:
        values = df[col].dropna().unique()[:5]
        print(f"    Exemples: {values}")

print("\n📊 RÉSUMÉ STATISTIQUE:")
print("="*50)
print(f"• Valeurs totales: {df.shape[0] * df.shape[1]:,}")
print(f"• Valeurs manquantes: {df.isnull().sum().sum():,}")
print(f"• Pourcentage de valeurs manquantes: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")

# Vérifier les colonnes numériques
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"• Colonnes numériques: {len(numeric_cols)}")
if len(numeric_cols) > 0:
    print(f"  → {list(numeric_cols)}")

# Vérifier les colonnes catégorielles
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"• Colonnes catégorielles: {len(categorical_cols)}")
if len(categorical_cols) > 0:
    print(f"  → {list(categorical_cols)}")

# ---------- 3. NETTOYAGE AUTOMATIQUE COMPLET ----------
print("\n" + "="*80)
print("NETTOYAGE AUTOMATIQUE DES DONNÉES")
print("="*80)

df_clean = df.copy()
cleaning_log = []

# 3.1 Conversion des types de données - CORRECTION
print("\n🔄 CONVERSION DES TYPES DE DONNÉES:")

# Conversion manuelle de is_question (car c'est important pour l'analyse)
if 'is_question' in df_clean.columns:
    # Créer un mapping explicite
    mapping = {'Oui': 1, 'OUI': 1, 'oui': 1, 'Yes': 1, 'yes': 1, 'TRUE': 1, 'True': 1, 'true': 1, '1': 1, 1: 1, True: 1}
    df_clean['is_question'] = df_clean['is_question'].astype(str).map(mapping)
    df_clean['is_question'] = df_clean['is_question'].fillna(0).astype(int)
    print(f"  ✓ is_question converti en numérique (0/1)")
    cleaning_log.append("✓ is_question converti en booléen (0/1)")

# Ne pas convertir message et message_preview en booléen (ce sont des textes)
# Garder les colonnes de texte comme object

# 3.2 Gestion des valeurs manquantes
print("\n🔍 GESTION DES VALEURS MANQUANTES:")
missing_before = df_clean.isnull().sum().sum()

if missing_before == 0:
    print("  ✓ Aucune valeur manquante détectée")
else:
    for col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            if df_clean[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                fill_value = df_clean[col].median()
                df_clean[col].fillna(fill_value, inplace=True)
                print(f"  ✓ {col}: {missing_count} valeurs remplacées par median ({fill_value:.2f})")
            else:
                fill_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                df_clean[col].fillna(fill_value, inplace=True)
                print(f"  ✓ {col}: {missing_count} valeurs remplacées par mode ('{fill_value}')")

missing_after = df_clean.isnull().sum().sum()
if missing_before > 0:
    cleaning_log.append(f"✓ Valeurs manquantes traitées: {missing_before} → {missing_after}")

# 3.3 Détection et traitement des outliers
print("\n🎯 DÉTECTION ET TRAITEMENT DES OUTLIERS:")

def treat_outliers_iqr(df, column):
    """Traite les outliers avec méthode IQR"""
    if df[column].dtype in ['int64', 'float64']:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:  # Éviter division par zéro
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Compter les outliers
            n_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            
            # Winsorizing (remplacement par les bornes)
            if n_outliers > 0:
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
                
                return n_outliers, lower_bound, upper_bound
    
    return 0, None, None

outlier_log = []
numeric_cols_for_outliers = ['message_length', 'engagement_score', 'turn_number']

for col in numeric_cols_for_outliers:
    if col in df_clean.columns:
        n_outliers, lower, upper = treat_outliers_iqr(df_clean, col)
        if n_outliers > 0:
            outlier_log.append(f"  • {col}: {n_outliers} outliers traités")
            print(f"  ✓ {col}: {n_outliers} outliers corrigés")

if not outlier_log:
    print("  Aucun outlier détecté nécessitant un traitement")

# 3.4 Normalisation des colonnes numériques (sauf strength qui a peu de valeurs)
print("\n📏 NORMALISATION DES DONNÉES:")
# Ne pas normaliser strength car il a seulement 3 valeurs
numeric_cols_to_normalize = [col for col in ['message_length', 'turn_number', 'engagement_score'] 
                            if col in df_clean.columns]

if len(numeric_cols_to_normalize) > 0:
    # Copier les valeurs originales de strength avant normalisation
    if 'strength' in df_clean.columns:
        df_clean['strength_original'] = df_clean['strength'].copy()
    
    scaler = MinMaxScaler()
    df_clean[numeric_cols_to_normalize] = scaler.fit_transform(df_clean[numeric_cols_to_normalize])
    cleaning_log.append(f"✓ {len(numeric_cols_to_normalize)} colonnes normalisées")
    print(f"  ✓ {len(numeric_cols_to_normalize)} colonnes numériques normalisées")
    print(f"  → Colonnes normalisées: {numeric_cols_to_normalize}")

# 3.5 Encodage des variables catégorielles
print("\n🔠 ENCODAGE DES VARIABLES CATÉGORIELLES:")
label_encoders = {}

# Colonnes catégorielles importantes à encoder
categorical_to_encode = ['speaker', 'strength_category', 'length_category']

for col in categorical_to_encode:
    if col in df_clean.columns:
        if df_clean[col].nunique() <= 20:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
            cleaning_log.append(f"✓ {col} encodé")
            print(f"  ✓ {col} encodé ({df_clean[col].nunique()} catégories)")
        else:
            print(f"  ⚠ {col} a trop de catégories ({df_clean[col].nunique()}) - non encodé")

# Ne pas encoder les colonnes de texte (_id, message_id, message_preview, message, processed_date)

# Affichage du résumé du nettoyage
print("\n" + "="*50)
print("RÉSUMÉ DU NETTOYAGE")
print("="*50)
for log in cleaning_log:
    print(f"  {log}")

print(f"\n📊 Après nettoyage:")
print(f"  • Shape: {df_clean.shape}")
print(f"  • Valeurs manquantes: {df_clean.isnull().sum().sum()}")
print(f"  • Types de données:")
for dtype in df_clean.dtypes.unique():
    count = sum(df_clean.dtypes == dtype)
    print(f"    - {dtype}: {count} colonnes")

# ---------- 4. CRÉATION DE FEATURES AVANCÉES ----------
print("\n" + "="*80)
print("CRÉATION DE FEATURES AVANCÉES")
print("="*80)

features_log = []
original_feature_count = len(df_clean.columns)

# 4.1 Vérifier la présence des colonnes nécessaires
print("\n🔍 VÉRIFICATION DES COLONNES DISPONIBLES:")

required_cols = ['strength', 'message_length', 'turn_number', 'is_question', 'engagement_score']
available_cols = []
for col in required_cols:
    if col in df_clean.columns:
        available_cols.append(col)
        print(f"  ✓ {col} disponible")
    else:
        print(f"  ✗ {col} non disponible")

print(f"\n✨ CRÉATION DES FEATURES AVEC {len(available_cols)} COLONNES DISPONIBLES:")

# 4.2 Features basées sur strength
if 'strength' in df_clean.columns:
    # Strength a seulement 3 valeurs (-1, 0, 1), donc on crée des features simples
    df_clean['strength_abs'] = np.abs(df_clean['strength'])
    df_clean['strength_positive'] = (df_clean['strength'] > 0).astype(int)
    df_clean['strength_negative'] = (df_clean['strength'] < 0).astype(int)
    df_clean['strength_neutral'] = (df_clean['strength'] == 0).astype(int)
    
    # Création d'une variable cible binaire (positive vs non-positive)
    df_clean['strength_binary'] = (df_clean['strength'] > 0).astype(int)
    
    features_log.append("✓ Features strength créées")
    print("  ✓ Features strength créées")

# 4.3 Features basées sur message_length
if 'message_length' in df_clean.columns:
    # Catégorisation de la longueur (basée sur les quantiles)
    try:
        # Utiliser des bins fixes au lieu de qcut qui peut échouer
        if df_clean['message_length'].max() > 0:
            bins = [0, 50, 100, 200, 500, float('inf')]
            labels = ['very_short', 'short', 'medium', 'long', 'very_long']
            df_clean['length_category_numeric'] = pd.cut(df_clean['message_length'], 
                                                       bins=bins, labels=range(len(labels)))
            features_log.append("✓ Catégorisation longueur créée")
            print("  ✓ Catégorisation longueur créée")
    except Exception as e:
        print(f"  ⚠ Erreur catégorisation longueur: {str(e)}")
    
    # Log de la longueur (ajouter 1 pour éviter log(0))
    df_clean['length_log'] = np.log1p(df_clean['message_length'])
    print("  ✓ Log longueur créé")

# 4.4 Features temporelles
if 'turn_number' in df_clean.columns:
    # Différence entre tours consécutifs
    df_clean['turn_delta'] = df_clean['turn_number'].diff().fillna(0)
    
    # Indicateur de changement de tour
    df_clean['turn_changed'] = (df_clean['turn_delta'] != 0).astype(int)
    
    features_log.append("✓ Features temporelles créées")
    print("  ✓ Features temporelles créées")

# 4.5 Features d'interaction
if all(col in df_clean.columns for col in ['strength', 'message_length']):
    # Ratio force/longueur
    df_clean['strength_per_length'] = df_clean['strength'] / (df_clean['message_length'] + 1)
    
    # Interaction
    df_clean['strength_x_length'] = df_clean['strength'] * df_clean['message_length']
    
    features_log.append("✓ Features d'interaction créées")
    print("  ✓ Features d'interaction créées")

# 4.6 Features avec engagement_score
if 'engagement_score' in df_clean.columns:
    # Normaliser l'engagement score s'il ne l'est pas déjà
    if df_clean['engagement_score'].max() > 1 or df_clean['engagement_score'].min() < 0:
        df_clean['engagement_normalized'] = (df_clean['engagement_score'] - df_clean['engagement_score'].min()) / \
                                           (df_clean['engagement_score'].max() - df_clean['engagement_score'].min())
    
    # Interaction avec strength
    if 'strength' in df_clean.columns:
        df_clean['strength_engagement'] = df_clean['strength'] * df_clean['engagement_score']
    
    features_log.append("✓ Features engagement créées")
    print("  ✓ Features engagement créées")

# 4.7 Features par speaker
if 'speaker' in df_clean.columns:
    # Statistiques par speaker
    if 'strength' in df_clean.columns:
        df_clean['speaker_strength_mean'] = df_clean.groupby('speaker')['strength'].transform('mean')
        df_clean['speaker_strength_std'] = df_clean.groupby('speaker')['strength'].transform('std').fillna(0)
    
    if 'message_length' in df_clean.columns:
        df_clean['speaker_length_mean'] = df_clean.groupby('speaker')['message_length'].transform('mean')
    
    features_log.append("✓ Features par speaker créées")
    print("  ✓ Features par speaker créées")

# 4.8 Features avec is_question
if 'is_question' in df_clean.columns:
    # Pourcentage de questions dans une fenêtre glissante
    if len(df_clean) > 10:
        df_clean['question_rolling_mean'] = df_clean['is_question'].rolling(window=10, min_periods=1).mean()
    
    # Interaction avec speaker
    if 'speaker' in df_clean.columns:
        df_clean['speaker_question'] = df_clean['speaker'] * df_clean['is_question']
    
    features_log.append("✓ Features questions créées")
    print("  ✓ Features questions créées")

# Résumé final des features
new_feature_count = len(df_clean.columns) - original_feature_count
print(f"\n📈 CRÉATION DE FEATURES TERMINÉE:")
print(f"  • Features originales: {original_feature_count}")
print(f"  • Features créées: {new_feature_count}")
print(f"  • Total features: {len(df_clean.columns)}")

if new_feature_count > 0:
    print(f"\n🎯 NOUVELLES FEATURES (premières 10):")
    # Identifier les nouvelles features (celles qui n'étaient pas dans df)
    original_cols_set = set(df.columns)
    new_features = [col for col in df_clean.columns if col not in original_cols_set]
    
    for i, feat in enumerate(new_features[:10], 1):
        print(f"  {i:2d}. {feat}")
    if len(new_features) > 10:
        print(f"  ... et {len(new_features) - 10} autres")

# ---------- 5. ANALYSE STATISTIQUE COMPLÈTE ----------
print("\n" + "="*80)
print("ANALYSE STATISTIQUE COMPLÈTE")
print("="*80)

# 5.1 Statistiques descriptives
print("\n📊 STATISTIQUES DESCRIPTIVES:")

# Colonnes principales à analyser
main_cols = ['strength', 'message_length', 'turn_number', 'engagement_score', 'is_question']
main_cols = [col for col in main_cols if col in df_clean.columns]

if main_cols:
    stats_df = df_clean[main_cols].describe().T
    # Sélectionner les statistiques pertinentes
    if 'count' in stats_df.columns:
        stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    print("\nStatistiques des variables principales:")
    print(stats_df.round(3))

# 5.2 Distribution de strength
if 'strength' in df_clean.columns:
    print(f"\n📈 DISTRIBUTION DE STRENGTH:")
    strength_counts = df_clean['strength'].value_counts().sort_index()
    for val, count in strength_counts.items():
        percent = (count / len(df_clean)) * 100
        label = {-1: 'Négatif', 0: 'Neutre', 1: 'Positif'}.get(val, val)
        print(f"  • {label}: {count} messages ({percent:.1f}%)")

# 5.3 Analyse par speaker
if 'speaker' in df_clean.columns:
    print(f"\n🎭 ANALYSE PAR SPEAKER:")
    
    # Décoder les valeurs de speaker pour l'affichage
    speaker_mapping = {0: 'usr', 1: 'sys'} if 'speaker' in label_encoders else {}
    
    for speaker_code in sorted(df_clean['speaker'].unique()):
        speaker_name = speaker_mapping.get(speaker_code, speaker_code)
        speaker_data = df_clean[df_clean['speaker'] == speaker_code]
        
        print(f"\n  {speaker_name}:")
        print(f"    • Messages: {len(speaker_data)} ({len(speaker_data)/len(df_clean)*100:.1f}%)")
        
        if 'strength' in df_clean.columns:
            print(f"    • Strength moyen: {speaker_data['strength'].mean():.3f}")
            print(f"    • Strength médian: {speaker_data['strength'].median():.3f}")
        
        if 'is_question' in df_clean.columns:
            question_percent = speaker_data['is_question'].mean() * 100
            print(f"    • Questions: {question_percent:.1f}%")
        
        if 'message_length' in df_clean.columns:
            print(f"    • Longueur moyenne: {speaker_data['message_length'].mean():.1f}")

# 5.4 Corrélations
print("\n🔗 CORRÉLATIONS ENTRE VARIABLES:")
correlation_cols = [col for col in ['strength', 'message_length', 'turn_number', 'engagement_score', 'is_question'] 
                   if col in df_clean.columns]

if len(correlation_cols) >= 2:
    corr_matrix = df_clean[correlation_cols].corr()
    print("\nMatrice de corrélation:")
    print(corr_matrix.round(3))
    
    # Identifier les corrélations significatives
    print("\nCorrélations notables (|r| > 0.1):")
    significant_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.1:
                significant_correlations.append((corr_matrix.columns[i], corr_matrix.columns[j], corr))
    
    if significant_correlations:
        for var1, var2, corr in significant_correlations:
            direction = "positive" if corr > 0 else "négative"
            print(f"  • {var1} ↔ {var2}: r = {corr:.3f} ({direction})")
    else:
        print("  Aucune corrélation notable détectée")

# ---------- 6. VISUALISATIONS AMÉLIORÉES ----------
print("\n" + "="*80)
print("CRÉATION DES VISUALISATIONS")
print("="*80)

# Création d'un dossier pour les visualisations
os.makedirs('visualizations', exist_ok=True)
print("📁 Dossier 'visualizations/' créé ou déjà existant")

# 6.1 Vérifier les colonnes disponibles pour les visualisations
print("\n🔍 COLONNES DISPONIBLES POUR VISUALISATION:")
viz_cols_available = []
for col in ['strength', 'message_length', 'speaker', 'turn_number', 'engagement_score']:
    if col in df_clean.columns:
        viz_cols_available.append(col)
        print(f"  ✓ {col} disponible")

if len(viz_cols_available) < 2:
    print("⚠️  Pas assez de colonnes pour créer des visualisations significatives")
else:
    print(f"\n✨ CRÉATION DES VISUALISATIONS AVEC {len(viz_cols_available)} COLONNES")
    
    # Préparer les données pour les visualisations
    # Utiliser un échantillon si le dataset est trop grand
    if len(df_clean) > 10000:
        plot_sample = df_clean.sample(10000, random_state=42)
        print(f"  ⚠ Utilisation d'un échantillon de 10,000 points pour les visualisations")
    else:
        plot_sample = df_clean
    
    # 6.2 Dashboard principal
    print("\n🎨 CRÉATION DU DASHBOARD PRINCIPAL...")
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('DASHBOARD ANALYTIQUE - ANALYSE CONVERSATIONNELLE AUGESC', 
                fontsize=20, fontweight='bold', y=1.02)
    
    # Grille 2x3
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
    
    # 6.2.1 Distribution de strength (bar plot au lieu d'histogramme)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'strength' in df_clean.columns:
        strength_counts = df_clean['strength'].value_counts().sort_index()
        colors = ['red', 'gray', 'green'] if len(strength_counts) == 3 else None
        bars = ax1.bar(range(len(strength_counts)), strength_counts.values, color=colors)
        ax1.set_title('Distribution de la Force Émotionnelle', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Strength')
        ax1.set_ylabel('Nombre de messages')
        ax1.set_xticks(range(len(strength_counts)))
        ax1.set_xticklabels(['Négatif', 'Neutre', 'Positif'][:len(strength_counts)])
        
        # Ajouter les comptes sur les barres
        for bar, count in zip(bars, strength_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    # 6.2.2 Strength par speaker
    ax2 = fig.add_subplot(gs[0, 1])
    if 'speaker' in df_clean.columns and 'strength' in df_clean.columns:
        # Décoder les speakers pour l'affichage
        if 'speaker' in label_encoders:
            speaker_labels = {0: 'usr', 1: 'sys'}
            plot_data = plot_sample.copy()
            plot_data['speaker_label'] = plot_data['speaker'].map(speaker_labels)
            hue = 'speaker_label'
        else:
            hue = 'speaker'
        
        sns.boxplot(data=plot_data, x=hue, y='strength', palette='Set2', ax=ax2)
        ax2.set_title('Strength par Speaker', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Speaker')
        ax2.set_ylabel('Strength')
        ax2.grid(True, alpha=0.2, axis='y')
    
    # 6.2.3 Relation strength vs longueur
    ax3 = fig.add_subplot(gs[0, 2])
    if all(col in df_clean.columns for col in ['strength', 'message_length']):
        scatter = ax3.scatter(plot_sample['message_length'], plot_sample['strength'], 
                             alpha=0.3, s=10, c=plot_sample['strength'], cmap='coolwarm')
        ax3.set_title('Relation: Longueur vs Strength', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Longueur du Message')
        ax3.set_ylabel('Strength')
        ax3.grid(True, alpha=0.2)
    
    # 6.2.4 Heatmap de corrélation
    ax4 = fig.add_subplot(gs[1, 0])
    if len(correlation_cols) >= 2:
        corr_matrix = df_clean[correlation_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax4)
        ax4.set_title('Matrice de Corrélation', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.tick_params(axis='y', rotation=0)
    
    # 6.2.5 Distribution de la longueur
    ax5 = fig.add_subplot(gs[1, 1])
    if 'message_length' in df_clean.columns:
        sns.histplot(data=plot_sample, x='message_length', kde=True, bins=50, 
                    color='purple', ax=ax5)
        ax5.axvline(plot_sample['message_length'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Moyenne: {plot_sample["message_length"].mean():.1f}')
        ax5.set_title('Distribution de la Longueur', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Longueur')
        ax5.set_ylabel('Fréquence')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.2)
    
    # 6.2.6 Questions par speaker
    ax6 = fig.add_subplot(gs[1, 2])
    if 'speaker' in df_clean.columns and 'is_question' in df_clean.columns:
        question_by_speaker = df_clean.groupby('speaker')['is_question'].mean() * 100
        
        # Décoder pour l'affichage
        if 'speaker' in label_encoders:
            speaker_labels = {0: 'usr', 1: 'sys'}
            question_by_speaker.index = question_by_speaker.index.map(speaker_labels)
        
        bars = ax6.bar(range(len(question_by_speaker)), question_by_speaker.values, 
                      color=['blue', 'orange'])
        ax6.set_title('Pourcentage de Questions par Speaker', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Speaker')
        ax6.set_ylabel('% de Questions')
        ax6.set_xticks(range(len(question_by_speaker)))
        ax6.set_xticklabels(question_by_speaker.index)
        
        # Ajouter les pourcentages
        for bar, percent in zip(bars, question_by_speaker.values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{percent:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/dashboard_principal.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    print("✓ Dashboard principal sauvegardé: visualizations/dashboard_principal.png")
    
    # 6.3 Visualisations supplémentaires
    print("\n🎨 CRÉATION DE VISUALISATIONS SUPPLÉMENTAIRES...")
    
    # 6.3.1 Évolution temporelle de strength
    fig_temp, ax_temp = plt.subplots(figsize=(12, 6))
    if 'turn_number' in df_clean.columns and 'strength' in df_clean.columns:
        # Prendre un échantillon pour éviter le surpeuplement
        temp_sample = df_clean.sort_values('turn_number')
        if len(temp_sample) > 5000:
            temp_sample = temp_sample.sample(5000, random_state=42).sort_values('turn_number')
        
        ax_temp.scatter(temp_sample['turn_number'], temp_sample['strength'], 
                       alpha=0.3, s=10, c=temp_sample['strength'], cmap='coolwarm')
        ax_temp.set_title('Évolution de Strength par Tour', fontsize=14, fontweight='bold')
        ax_temp.set_xlabel('Tour de Conversation')
        ax_temp.set_ylabel('Strength')
        ax_temp.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig('visualizations/evolution_temporelle.png', dpi=300, bbox_inches='tight')
        print("✓ Évolution temporelle sauvegardée")
    
    # 6.3.2 Violin plot strength par speaker
    fig_violin, ax_violin = plt.subplots(figsize=(10, 6))
    if 'speaker' in df_clean.columns and 'strength' in df_clean.columns:
        sns.violinplot(data=plot_sample, x='speaker', y='strength', 
                      palette='muted', ax=ax_violin)
        ax_violin.set_title('Distribution de Strength par Speaker', 
                          fontsize=14, fontweight='bold')
        ax_violin.set_xlabel('Speaker')
        ax_violin.set_ylabel('Strength')
        plt.tight_layout()
        plt.savefig('visualizations/violinplot_strength.png', dpi=300, bbox_inches='tight')
        print("✓ Violin plot sauvegardé")
    
    # 6.3.3 Engagement score vs strength
    if all(col in df_clean.columns for col in ['strength', 'engagement_score']):
        fig_eng, ax_eng = plt.subplots(figsize=(10, 6))
        scatter = ax_eng.scatter(plot_sample['engagement_score'], plot_sample['strength'], 
                                alpha=0.5, s=20, c=plot_sample['strength'], cmap='coolwarm')
        ax_eng.set_title('Engagement Score vs Strength', fontsize=14, fontweight='bold')
        ax_eng.set_xlabel('Engagement Score')
        ax_eng.set_ylabel('Strength')
        ax_eng.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig('visualizations/engagement_vs_strength.png', dpi=300, bbox_inches='tight')
        print("✓ Engagement vs strength sauvegardé")
    
    plt.close('all')  # Fermer toutes les figures
    print("✅ Toutes les visualisations ont été créées et sauvegardées")

# ---------- 7. MODÈLES ML COMPLETS ----------
print("\n" + "="*80)
print("ENTRAÎNEMENT DES MODÈLES DE MACHINE LEARNING")
print("="*80)

# 7.1 Préparation des données pour ML
print("\n🎯 PRÉPARATION DES DONNÉES POUR ML:")

# Vérifier si nous avons une variable cible
if 'strength_binary' not in df_clean.columns:
    # Créer une variable cible binaire basée sur strength positif
    if 'strength' in df_clean.columns:
        df_clean['strength_binary'] = (df_clean['strength'] > 0).astype(int)
        target = 'strength_binary'
        print(f"✓ Variable cible créée: strength_binary (positive vs non-positive)")
    else:
        print("❌ Impossible de créer une variable cible: colonne 'strength' manquante")
        target = None
else:
    target = 'strength_binary'
    print("✓ Variable cible existante: strength_binary")

if target is not None:
    # Vérifier la distribution de la variable cible
    y = df_clean[target]
    print(f"\nDistribution de la variable cible ({target}):")
    value_counts = y.value_counts()
    for val, count in value_counts.items():
        percent = (count / len(y)) * 100
        label = 'Positif' if val == 1 else 'Non-positif'
        print(f"  {label}: {count} échantillons ({percent:.1f}%)")
    
    # Sélectionner les features
    # Exclure les colonnes non pertinentes
    exclude_cols = [target, '_id', 'message_id', 'message_preview', 'message', 
                   'processed_date', 'strength_original']
    exclude_cols = [col for col in exclude_cols if col in df_clean.columns]
    
    # Sélectionner les features numériques
    features = [col for col in df_clean.columns 
               if col not in exclude_cols 
               and df_clean[col].dtype in [np.int64, np.float64, np.int32, np.float32]
               and df_clean[col].nunique() > 1]
    
    print(f"\nFeatures sélectionnées: {len(features)}")
    if len(features) > 10:
        print(f"  Top 10 features: {features[:10]}")
        print(f"  ... et {len(features) - 10} autres")
    else:
        print(f"  Features: {features}")
    
    if len(features) < 2:
        print("⚠️  Pas assez de features pour l'entraînement ML")
    else:
        X = df_clean[features].copy()
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"\nDivision des données:")
        print(f"  • Train: {X_train.shape[0]} échantillons, {X_train.shape[1]} features")
        print(f"  • Test:  {X_test.shape[0]} échantillons, {X_test.shape[1]} features")
        
        # 7.2 Définition des modèles
        print("\n🤖 CONFIGURATION DES MODÈLES:")
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Gaussian Naive Bayes': GaussianNB()
        }
        
        print(f"✓ {len(models)} modèles configurés")
        
        # 7.3 Entraînement et évaluation
        print("\n🚀 ENTRAÎNEMENT DES MODÈLES:")
        print("="*60)
        
        results = []
        feature_importances = {}
        
        for name, model in models.items():
            print(f"\n{'='*40}")
            print(f"Modèle: {name}")
            print(f"{'='*40}")
            
            try:
                # Entraînement
                model.fit(X_train, y_train)
                
                # Prédictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Métriques de base
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                model_results = {
                    'Modèle': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1
                }
                
                # Métriques avancées si disponibles
                if y_pred_proba is not None:
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                        model_results['ROC-AUC'] = roc_auc
                    except:
                        pass
                
                # Importance des features
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(10)
                    feature_importances[name] = importance_df
                
                results.append(model_results)
                
                # Affichage des résultats
                print(f"Accuracy:  {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall:    {recall:.4f}")
                print(f"F1-Score:  {f1:.4f}")
                if 'ROC-AUC' in model_results:
                    print(f"ROC-AUC:   {model_results['ROC-AUC']:.4f}")
                
            except Exception as e:
                print(f"❌ Erreur: {str(e)}")
                continue
        
        # 7.4 Comparaison des modèles
        if results:
            print("\n" + "="*80)
            print("COMPARAISON DES PERFORMANCES")
            print("="*80)
            
            results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
            print("\nClassement des modèles:")
            print(results_df.to_string(index=False))
            
            # Sauvegarde des résultats
            results_df.to_csv('model_performance.csv', index=False)
            print("\n✓ Résultats sauvegardés dans 'model_performance.csv'")
            
            # 7.5 Visualisation des performances
            print("\n🎨 CRÉATION DES VISUALISATIONS ML...")
            
            fig_ml, axes_ml = plt.subplots(2, 2, figsize=(14, 10))
            fig_ml.suptitle('PERFORMANCE DES MODÈLES DE MACHINE LEARNING', 
                          fontsize=18, fontweight='bold', y=1.02)
            
            # Graphique 1: Accuracy par modèle
            axes_ml[0, 0].barh(results_df['Modèle'], results_df['Accuracy'], 
                              color=plt.cm.Set3(np.arange(len(results_df))))
            axes_ml[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
            axes_ml[0, 0].set_xlabel('Accuracy')
            axes_ml[0, 0].set_xlim([0, 1])
            for i, v in enumerate(results_df['Accuracy']):
                axes_ml[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
            
            # Graphique 2: Matrice de confusion du meilleur modèle
            best_model_name = results_df.iloc[0]['Modèle']
            best_model = models.get(best_model_name)
            if best_model:
                y_pred_best = best_model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred_best)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_ml[0, 1])
                axes_ml[0, 1].set_title(f'Matrice de Confusion - {best_model_name}', 
                                      fontsize=14, fontweight='bold')
                axes_ml[0, 1].set_xlabel('Prédit')
                axes_ml[0, 1].set_ylabel('Réel')
            
            # Graphique 3: Courbes ROC (si disponibles)
            axes_ml[1, 0].plot([0, 1], [0, 1], 'k--', label='Aléatoire', linewidth=1)
            roc_models = []
            for name, model in models.items():
                if hasattr(model, 'predict_proba'):
                    try:
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                        axes_ml[1, 0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', 
                                          linewidth=1.5)
                        roc_models.append(name)
                    except:
                        continue
            
            if roc_models:
                axes_ml[1, 0].set_title('Courbes ROC', fontsize=14, fontweight='bold')
                axes_ml[1, 0].set_xlabel('Taux de Faux Positifs')
                axes_ml[1, 0].set_ylabel('Taux de Vrais Positifs')
                axes_ml[1, 0].legend(loc='lower right', fontsize=8)
                axes_ml[1, 0].grid(True, alpha=0.2)
            
            # Graphique 4: Importance des features (meilleur modèle)
            if best_model_name in feature_importances and not feature_importances[best_model_name].empty:
                importance_df = feature_importances[best_model_name].head(10)
                axes_ml[1, 1].barh(importance_df['Feature'], importance_df['Importance'], 
                                  color='skyblue')
                axes_ml[1, 1].set_title(f'Top Features - {best_model_name}', 
                                      fontsize=14, fontweight='bold')
                axes_ml[1, 1].set_xlabel('Importance')
                axes_ml[1, 1].invert_yaxis()
                for i, v in enumerate(importance_df['Importance']):
                    axes_ml[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('visualizations/ml_performance.png', dpi=300, 
                       bbox_inches='tight', facecolor='white')
            print("✓ Visualisations ML sauvegardées: visualizations/ml_performance.png")
            
            # Sauvegarde du meilleur modèle
            if best_model:
                try:
                    with open('meilleur_modele.pkl', 'wb') as f:
                        pickle.dump(best_model, f)
                    print("✓ Meilleur modèle sauvegardé: 'meilleur_modele.pkl'")
                except:
                    print("⚠ Impossible de sauvegarder le modèle")
        else:
            print("❌ Aucun modèle n'a pu être entraîné")
else:
    print("❌ Impossible de procéder à l'analyse ML sans variable cible")

# ---------- 8. SAUVEGARDE ET RAPPORT ----------
print("\n" + "="*80)
print("SAUVEGARDE ET GÉNÉRATION DU RAPPORT")
print("="*80)

# 8.1 Sauvegarde des données enrichies
print("\n💾 SAUVEGARDE DES DONNÉES...")

df_clean.to_csv('augesc_data_clean.csv', index=False)
print("✓ Données nettoyées sauvegardées: 'augesc_data_clean.csv'")

# 8.2 Génération du rapport de synthèse
print("\n📋 GÉNÉRATION DU RAPPORT DE SYNTHÈSE...")

report = {
    "meta": {
        "date_analyse": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script_version": "3.0",
        "dataset_original": f"{df.shape[0]} lignes, {df.shape[1]} colonnes"
    },
    "nettoyage": {
        "valeurs_manquantes": int(missing_before),
        "outliers_traites": len(outlier_log),
        "colonnes_normalisees": len(numeric_cols_to_normalize) if 'numeric_cols_to_normalize' in locals() else 0,
        "colonnes_encodees": len(label_encoders)
    },
    "donnees_finales": {
        "lignes": len(df_clean),
        "colonnes": len(df_clean.columns),
        "features_crees": new_feature_count if 'new_feature_count' in locals() else 0
    },
    "statistiques": {},
    "machine_learning": {}
}

# Ajouter des statistiques
if 'strength' in df_clean.columns:
    strength_dist = df_clean['strength'].value_counts().to_dict()
    report["statistiques"]["strength"] = {
        "distribution": {str(k): int(v) for k, v in strength_dist.items()},
        "moyenne": float(df_clean['strength'].mean()),
        "positif": int((df_clean['strength'] > 0).sum()),
        "negatif": int((df_clean['strength'] < 0).sum()),
        "neutre": int((df_clean['strength'] == 0).sum())
    }

if 'message_length' in df_clean.columns:
    report["statistiques"]["message_length"] = {
        "moyenne": float(df_clean['message_length'].mean()),
        "ecart_type": float(df_clean['message_length'].std()),
        "min": float(df_clean['message_length'].min()),
        "max": float(df_clean['message_length'].max())
    }

# Ajouter l'analyse par speaker
if 'speaker' in df_clean.columns:
    speaker_stats = {}
    for speaker_code in df_clean['speaker'].unique():
        speaker_data = df_clean[df_clean['speaker'] == speaker_code]
        speaker_stats[str(speaker_code)] = {
            "count": int(len(speaker_data)),
            "percentage": float(len(speaker_data) / len(df_clean) * 100)
        }
    report["statistiques"]["speakers"] = speaker_stats

# Ajouter les résultats ML si disponibles
if 'results_df' in locals() and not results_df.empty:
    report["machine_learning"] = {
        "nombre_modeles": int(len(results_df)),
        "meilleur_modele": results_df.iloc[0]['Modèle'],
        "accuracy_meilleur": float(results_df.iloc[0]['Accuracy']),
        "top_3_modeles": results_df.head(3)[['Modèle', 'Accuracy']].to_dict('records')
    }

# Sauvegarder le rapport
with open('rapport_analyse_augesc.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("✓ Rapport généré: 'rapport_analyse_augesc.json'")

# 8.3 Résumé final
print("\n" + "="*80)
print("RÉSUMÉ FINAL DE L'ANALYSE")
print("="*80)

print(f"\n📊 DONNÉES:")
print(f"  • Messages analysés: {len(df_clean):,}")
print(f"  • Features finales: {len(df_clean.columns)}")

if 'strength' in df_clean.columns:
    print(f"\n📈 DISTRIBUTION STRENGTH:")
    pos = (df_clean['strength'] > 0).sum()
    neg = (df_clean['strength'] < 0).sum()
    neu = (df_clean['strength'] == 0).sum()
    total = len(df_clean)
    print(f"  • Positif: {pos:,} ({pos/total*100:.1f}%)")
    print(f"  • Négatif: {neg:,} ({neg/total*100:.1f}%)")
    print(f"  • Neutre: {neu:,} ({neu/total*100:.1f}%)")

if 'speaker' in df_clean.columns:
    print(f"\n🎭 RÉPARTITION PAR SPEAKER:")
    for speaker_code in sorted(df_clean['speaker'].unique()):
        count = (df_clean['speaker'] == speaker_code).sum()
        percent = count / len(df_clean) * 100
        speaker_name = 'usr' if speaker_code == 0 else 'sys' if speaker_code == 1 else str(speaker_code)
        print(f"  • {speaker_name}: {count:,} ({percent:.1f}%)")

if 'results_df' in locals() and not results_df.empty:
    print(f"\n🤖 PERFORMANCE ML:")
    print(f"  • Modèles testés: {len(results_df)}")
    print(f"  • Meilleur modèle: {results_df.iloc[0]['Modèle']}")
    print(f"  • Accuracy: {results_df.iloc[0]['Accuracy']:.1%}")
    if 'Precision' in results_df.columns:
        print(f"  • Precision: {results_df.iloc[0]['Precision']:.1%}")
    if 'Recall' in results_df.columns:
        print(f"  • Recall: {results_df.iloc[0]['Recall']:.1%}")

print(f"\n📁 FICHIERS GÉNÉRÉS:")
print("  • augesc_data_clean.csv - Données nettoyées")
print("  • rapport_analyse_augesc.json - Rapport de synthèse")
if os.path.exists('visualizations/'):
    vis_files = os.listdir('visualizations')
    print(f"  • visualizations/ - {len(vis_files)} visualisations")
if os.path.exists('model_performance.csv'):
    print("  • model_performance.csv - Performance des modèles")
if os.path.exists('meilleur_modele.pkl'):
    print("  • meilleur_modele.pkl - Meilleur modèle sauvegardé")

print(f"\n✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
print("="*80)

# 8.4 Export des insights pour l'utilisateur
print("\n🔍 INSIGHTS CLÉS:")
print("="*50)

if 'strength' in df_clean.columns:
    # Insight 1: Distribution de strength
    pos_pct = (df_clean['strength'] > 0).mean() * 100
    neg_pct = (df_clean['strength'] < 0).mean() * 100
    print(f"1. La force émotionnelle est principalement {'positive' if pos_pct > 50 else 'négative'}")
    print(f"   → Positif: {pos_pct:.1f}%, Négatif: {neg_pct:.1f}%")
    
    # Insight 2: Différence entre speakers
    if 'speaker' in df_clean.columns:
        speaker_strength = df_clean.groupby('speaker')['strength'].mean()
        if len(speaker_strength) >= 2:
            diff = abs(speaker_strength.iloc[0] - speaker_strength.iloc[1])
            if diff > 0.1:
                speaker_names = {0: 'usr', 1: 'sys'}
                stronger = 'usr' if speaker_strength.iloc[0] > speaker_strength.iloc[1] else 'sys'
                print(f"2. Le speaker '{stronger}' a une force émotionnelle plus élevée")
                print(f"   → Différence: {diff:.3f}")
    
    # Insight 3: Relation avec la longueur
    if 'message_length' in df_clean.columns:
        corr = df_clean['strength'].corr(df_clean['message_length'])
        if abs(corr) > 0.1:
            direction = "positive" if corr > 0 else "négative"
            print(f"3. Relation {direction} entre force et longueur des messages")
            print(f"   → Corrélation: {corr:.3f}")

print("\n🎯 RECOMMANDATIONS:")
print("="*50)
print("1. Analyser les messages avec force négative pour comprendre les problèmes")
print("2. Étudier les patterns des speakers pour optimiser les conversations")
print("3. Utiliser le modèle ML pour prédire la force émotionnelle en temps réel")
print("4. Surveiller l'engagement score pour améliorer la qualité des interactions")

print("\n" + "="*80)
print("📊 ANALYSE COMPLÈTE - PRÊT POUR LA PRISE DE DÉCISION!")
print("="*80)