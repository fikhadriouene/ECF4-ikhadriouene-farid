# ECF4 -- Détection de Fake News (NLP, TensorFlow & FastAPI)

## Objectif

L’objectif est de concevoir un système capable de prédire à partir d'un titre d’article, si on a affaire à une fake news ou une information réelle.

------------------------------------------------------------------------

# Etapes du projet

- Implémentation du pipeline NLP :
    - Nettoyage du texte (lowercase, suppression des caractères spéciaux)
    - Suppression des stopwords
    - Lemmatisation

- Vectorisation :
    - TF-IDF

- Modélisation :
    - Evaluation de plusieurs modèles (réseau dense sur TF-IDF, BiLSTM)
    - Versions améliorées des modèles

- Sélection du modèle :
    - Comparaison des performances
    - Choix du meilleur modèle

- Mise en production :
    - Développement d’une API REST avec FastAPI

------------------------------------------------------------------------

# Structure du projet

ECF4/

│

├── api/

│   └── main.py

│

├── data/

│   ├── news.csv

│   └── titles_clean.csv

│

├── models/

│   ├── best_model.keras

│   ├── best_model_ameliore.keras

│   ├── best_model_bilstm.keras

│   ├── best_model_bilstm_ameliore.keras

│   └── vectorizer.pkl

│

├── notebook/

│   └── ecf_fake_news.ipynb

│

├── docker-compose.yml

├── Dockerfile

├── requirements.txt

├── README_DOCKER.txt

├── sujet.md

├── Rapport.md

└── README.md

------------------------------------------------------------------------

# Description des dossiers et fichiers

# data :
 - news.csv : dataset brut
 - titles_clean.csv : dataset nettoyé

# notebook :
 - ecf_fake_news.ipynb : pipeline complet

# models :
 - modèles entraînés + vectorizer

# api :
 - main.py : API FastAPI



# Lancement
A la racine du projet exécuter :

  docker-compose up -d :

    - Chargement des librairies nécessaires
    - Exécution du notebook ecf_fake_news.ipynb pour la création du best_model et vectorizer
    - Exécution du main.py qui lance l'api

ou

- python -m venv venv
- venv\Scripts\activate
- pip install -r requirements.txt
- python -m spacy download en_core_web_sm
- Exécution du notebook ecf_fake_news.ipynb dans vscode ou Jupyter pour la création du best_model et vectorizer 
- uvicorn api.main:app --reload

Accéder à l'API via : 
http://127.0.0.1:8000/docs

------------------------------------------------------------------------

# Exemple

/predict

Reçoit un seul titre et retourne :

le titre reçu
le label prédit
le score de confiance

Entrée :
{ "title": "Scientists discover new treatment" }

Sortie :
{
  "title": "Scientists discover new treatment",
  "label": "FAKE",
  "confidence": 0.86
}

------------------------------------------------------------------------
------------------------------------------------------------------------

/predict/batch

Reçoit une liste de titres et retourne :

Entrée :
{
  "titles": [
    "Scientists discover new treatment",
    "Government announces major reform"
  ]
}

Sortie :
{
  "predictions": [
    {
      "title": "Scientists discover new treatment",
      "label": "FAKE",
      "confidence": 0.86
    },
    {
      "title": "Government announces major reform",
      "label": "FAKE",
      "confidence": 0.85
    }
  ]
}


# Modèle retenu

réseau dense sur TF-IDF

------------------------------------------------------------------------

# Améliorations

-Paramètre des modèles :
    - Diminution de la taille du vocabulaire
    - Augmentation du dropout
    - Diminution du nombre de neurones
    - Ajout de la régularisation L2  sur les couches dense

- transfert learning avec BERT
- data augmentation avec back translation 


