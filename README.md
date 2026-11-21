
# 🐜 Simulation de l'algorithme ACO

Une application interactive développée avec Streamlit pour illustrer
le fonctionnement de l'algorithme d'optimisation par colonies de fourmis
(Ant Colony Optimization).

## 🎯 Objectif

Simuler plusieurs fourmis artificielles qui recherchent une source de
nourriture à partir d'un nid.

Chaque fourmi prend ses décisions en fonction de :

- La quantité de phéromone.
- L'information heuristique.
- La probabilité de transition.
- La distance jusqu'à la destination.

Les chemins trouvés sont évalués et les phéromones sont mises à jour
afin de favoriser progressivement les chemins les plus courts.

## ✨ Fonctionnalités

- Environnement en grille.
- Nid, nourriture et obstacles.
- Plusieurs fourmis artificielles.
- Visualisation des chemins.
- Calcul des probabilités de transition.
- Matrice des phéromones.
- Évaporation et renforcement.
- Exécution étape par étape.
- Pseudocode de l'algorithme.

## 🛠️ Technologies

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib

## 🚀 Lancer l'application

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer Streamlit :

```bash
streamlit run app.py
```

## 🌐 Déploiement sur Streamlit Community Cloud

1. Ajouter les fichiers du projet dans un dépôt GitHub.
2. Ouvrir https://share.streamlit.io/
3. Se connecter avec GitHub.
4. Cliquer sur New app.
5. Sélectionner le dépôt et la branche.
6. Choisir `app.py` comme fichier principal.
7. Cliquer sur Deploy.

## 📐 Modèle mathématique

La probabilité de transition est :

Pij = (τij^α × ηij^β) /
      Σ(τil^α × ηil^β)

L'information heuristique est :

ηij = 1 / d(j, destination)

L'évaporation est :

τij = (1 - ρ) × τij

Le dépôt de phéromone est :

Δτ = Q / Lk

## 📚 Remarque

Cette application est un modèle pédagogique inspiré du comportement
collectif des fourmis. La phéromone est représentée par cellule afin
de faciliter la visualisation.
