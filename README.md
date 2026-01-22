# 📊 Dashboard d'Analyse Conversationnelle AUGESC

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Viz](https://img.shields.io/badge/Visualisation-Matplotlib%2FSeaborn-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 Aperçu

Ce projet implémente un dashboard complet d'analyse de données conversationnelles avec capacités avancées de Machine Learning. Le système analyse les conversations pour extraire des insights sur la force émotionnelle, l'engagement, et les patterns d'interaction.

## 🚀 Fonctionnalités

### 🔍 Analyse des Données
- **Nettoyage automatique** des données conversationnelles
- **Création de features** avancées (temporelles, d'interaction, par speaker)
- **Statistiques descriptives** complètes
- **Visualisations interactives** et exportables

### 🤖 Machine Learning
- **7 algorithmes ML** prêts à l'emploi :
  - Régression Logistique
  - Random Forest
  - Gradient Boosting
  - Decision Tree
  - K-Nearest Neighbors
  - Gaussian Naive Bayes
- **Évaluation automatique** avec métriques multiples
- **Sélection du meilleur modèle**
- **Importance des features**

### 📊 Visualisations
- Dashboard principal 2×3 avec 6 visualisations
- Heatmaps de corrélation
- Distributions par speaker
- Évolutions temporelles
- Performances des modèles ML

## 🛠 Technologies Utilisées

<div align="center">

### 🐍 Langages & Frameworks
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### 📈 Machine Learning
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### 📊 Visualisation
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-6C9BC9?style=for-the-badge)

### 📁 Gestion de Données
![JSON](https://img.shields.io/badge/JSON-000000?style=for-the-badge&logo=json&logoColor=white)
![CSV](https://img.shields.io/badge/CSV-239120?style=for-the-badge&logo=csv&logoColor=white)

</div>

## 📋 Structure du Projet

```
DATA-ANALYST-AUGESC/
│
├── 📄 augesc_export_ready.csv          # Données brutes d'entrée
├── 📄 augesc_data_clean.csv            # Données nettoyées
├── 📄 model_performance.csv            # Performances des modèles ML
├── 📄 rapport_analyse_augesc.json      # Rapport de synthèse JSON
├── 📄 meilleur_modele.pkl              # Meilleur modèle sauvegardé
│
├── 📁 visualizations/                  # Toutes les visualisations
│   ├── 📊 dashboard_principal.png
│   ├── 📈 evolution_temporelle.png
│   ├── 🎻 violinplot_strength.png
│   ├── 🔥 engagement_vs_strength.png
│   └── 🤖 ml_performance.png
│
└── 📄 README.md                        # Ce fichier
```

## 🔧 Installation

1. **Cloner le repository**
```bash
git clone https://github.com/Bourzguifatimazahra/DATA-ANALYST-AUGESC.git
cd DATA-ANALYST-AUGESC
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
```

## 🎯 Utilisation

### Exécution complète
```python
python augesc_analysis.py
```

### Fonctionnalités principales :

1. **Chargement automatique** des données
2. **Nettoyage intelligent** avec logging
3. **Création de features** avancées
4. **Analyse statistique** complète
5. **Entraînement ML** avec 7 algorithmes
6. **Génération de visualisations**
7. **Export des résultats**

## 📊 Sorties Générées

### 📈 Visualisations
- **Dashboard principal** (6 graphiques)
- **Heatmaps** de corrélation
- **Distributions** par speaker
- **Courbes ROC** pour les modèles ML
- **Importance des features**

### 📄 Fichiers Exportés
- `augesc_data_clean.csv` - Données enrichies
- `rapport_analyse_augesc.json` - Rapport structuré
- `model_performance.csv` - Comparaison des modèles
- `meilleur_modele.pkl` - Modèle ML sauvegardé

## 🔍 Insights Clés

Le système identifie automatiquement :
- **Distribution émotionnelle** (positif/négatif/neutre)
- **Différences entre speakers**
- **Corrélations** entre features
- **Patterns temporels**
- **Features les plus importantes** pour la prédiction

## 🎨 Personnalisation

### Modifier les paramètres ML
```python
# Dans la section "7.2 Définition des modèles"
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,      # Augmenter le nombre d'arbres
        max_depth=15,          # Augmenter la profondeur
        random_state=42
    ),
    # ... autres modèles
}
```

### Ajouter de nouvelles features
```python
# Dans la section "4. CRÉATION DE FEATURES AVANCÉES"
df_clean['nouvelle_feature'] = df_clean['col1'] * df_clean['col2']
```

## 📊 Métriques de Performance

Le système évalue les modèles avec :
- ✅ **Accuracy** - Précision globale
- ✅ **Precision** - Pertinence des positifs prédits
- ✅ **Recall** - Capacité à détecter les vrais positifs
- ✅ **F1-Score** - Moyenne harmonique
- ✅ **ROC-AUC** - Performance générale

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Crée une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvre une Pull Request

## ✨ Auteur

**Fatima Zahra Bourzgui**
- GitHub: [@Bourzguifatimazahra](https://github.com/Bourzguifatimazahra)
- Projet: [DATA-ANALYST-AUGESC](https://github.com/Bourzguifatimazahra/DATA-ANALYST-AUGESC)

## 🙏 Remerciements

- [Scikit-learn](https://scikit-learn.org/) pour les outils ML
- [Matplotlib](https://matplotlib.org/) et [Seaborn](https://seaborn.pydata.org/) pour la visualisation
- [Pandas](https://pandas.pydata.org/) pour la manipulation de données

---

<div align="center">
  
**🌟 Si ce projet vous est utile, pensez à lui donner une étoile !**

[![Star History Chart](https://api.star-history.com/svg?repos=Bourzguifatimazahra/DATA-ANALYST-AUGESC&type=Date)](https://star-history.com/#Bourzguifatimazahra/DATA-ANALYST-AUGESC&Date)

</div>
