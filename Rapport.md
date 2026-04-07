# Rapport ECF4 – Détection de Fake News

## 1. Contexte

Ce projet s’inscrit dans le cadre de l’ECF « Concepteur Développeur en Intelligence Artificielle ».

L’objectif est de concevoir un système capable de prédire si un titre d’article est une fake news ou une information réelle.

---

## 2. Données

Dataset utilisé :
- news.csv : dataset brut
- titles_clean.csv : dataset après nettoyage

Variable utilisée :
- title

Variable cible :
- 0 : FAKE
- 1 : REAL

---

## 3. Prétraitement

Les étapes suivantes ont été appliquées :

- passage en minuscules
- suppression des caractères spéciaux
- suppression des stopwords
- lemmatisation

Objectif :
Réduire le bruit et normaliser les données textuelles.

---

## 4. Vectorisation

Méthode utilisée :
- TF-IDF

Fichier :
- vectorizer.pkl

Le texte est transformé en vecteurs numériques exploitables par les modèles.

---

## 5. Modélisation

Plusieurs modèles ont été testés :

### 5.1 Dense Neural Network
- modèle simple
- rapide à entraîner

### 5.2 BiLSTM
- modèle basé sur les séquences
- meilleure prise en compte du contexte

---

## 6. Choix du modèle

Le modèle retenu est celui offrant le meilleur compromis entre :
- performance
- temps d’entraînement
- complexité

Le modèle Dense amélioré a été privilégié.

---

## 7. Évaluation

Métriques utilisées :

- accuracy
- precision
- recall
- f1-score

Analyse :

- faux positifs : fake classée comme real
- faux négatifs : real classée comme fake

Ces erreurs permettent d’identifier les limites du modèle.

---

## 8. Mise en production

Une API REST a été développée avec FastAPI.

Fichier :
- api/main.py

Endpoints :

- GET /health
- POST /predict
- POST /predict/batch

Le modèle et le vectorizer sont chargés au démarrage.

---

## 9. Exemple de prédiction

Entrée :

{
  "title": "Scientists discover new treatment"
}

Sortie :

{
  "title": "Scientists discover new treatment",
  "label": "REAL",
  "confidence": 0.87
}

---

## 10. Limites

- utilisation uniquement du titre (pas du contenu complet)
- dépendance au dataset
- difficulté sur les titres ambigus

---

## 11. Améliorations possibles

- utilisation de modèles transformers (BERT)
- prise en compte du texte complet
-Paramètre des modèles :
    - Diminution de la taille du vocabulaire
    - Augmentation du dropout
    - Diminution du nombre de neurones
    - Ajout de la régularisation L2  sur les couches dense
- transfert learning avec BERT
- data augmentation avec back translation 


## Bonus : Versions améliorées
- best_model_ameliore.keras => diminution de l'overfitting
- best_model_bilstm_ameliore.keras => pas de diminution de l'overfitting
---


