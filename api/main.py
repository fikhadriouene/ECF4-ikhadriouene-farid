from __future__ import annotations

import logging
import re
from typing import List

import joblib
import nltk
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

english_stopwords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Configuration des logs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Création de l'application FastAPI

app = FastAPI(
    title="Fake News Detection API",
    description="API REST de détection de fake news à partir de titres d'articles.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Chargement du vectorizer et du modèle au démarrage

try:
    tfidf_vectorizer = joblib.load("./models/vectorizer.pkl")
    fake_news_model = load_model("./models/best_model.keras")
    logger.info("Modèle et vectorizer chargés avec succès.")
except Exception as error:
    logger.exception("Erreur de chargement des ressources")
    fake_news_model = None
    tfidf_vectorizer = None


# Schémas Pydantic

class PredictRequest(BaseModel):
    title: str = Field(
        ...,
        description="Titre d'article à analyser",
        examples=["Scientists discover new treatment"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Scientists discover new treatment"
            }
        }
    )


class PredictResponse(BaseModel):
    title: str = Field(..., description="Titre reçu par l'API")
    label: str = Field(..., description="Classe prédite : REAL ou FAKE", examples=["REAL"])
    confidence: float = Field(..., description="Score de confiance entre 0 et 1", examples=[0.87])


class BatchRequest(BaseModel):
    titles: List[str] = Field(
        ...,
        description="Liste de titres à analyser",
        examples=[["Scientists discover new treatment", "Government announces major reform"]],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "titles": [
                    "Scientists discover new treatment",
                    "Government announces major reform"
                ]
            }
        }
    )


class BatchResponse(BaseModel):
    predictions: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model: str



# Fonction de nettoyage du texte 
def clean_title(raw_title: str) -> str:

    title_lowercase = raw_title.lower().strip()
    title_without_urls = re.sub(r"http\S+|www\S+", " ", title_lowercase)
    title_letters_only = re.sub(r"[^a-zA-Z\s']", " ", title_without_urls)

    tokens = word_tokenize(title_letters_only)

    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in english_stopwords and len(token) > 1
    ]

    cleaned_title = " ".join(cleaned_tokens)
    return cleaned_title



# Validation métier d'un titre

def validate_title(title: str) -> None:
    if title.strip() == "" :
        raise HTTPException(
            status_code=422,
            detail="Le titre ne peut pas être vide ou composé uniquement d'espaces."
        )

    if len(title) > 300:
        raise HTTPException(
            status_code=400,
            detail="Le titre dépasse 300 caractères."
        )



# Vérification du chargement du modèle et du vectorizer

def verify_model_vectorizer_loaded() -> None:

    if fake_news_model is None or tfidf_vectorizer is None:
        raise HTTPException(
            # status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status_code=500,
            detail="Le modèle ou le vectorizer n'est pas disponible."
        )



# Fonction principale de prédiction

def predict_title(title: str) -> PredictResponse:

    verify_model_vectorizer_loaded()

    cleaned_title = clean_title(title)

    # Transformation du texte en vecteur numérique
    title_vector = tfidf_vectorizer.transform([cleaned_title]).toarray()
    print(tfidf_vectorizer)
    # Prédiction du score par le modèle
    raw_prediction = fake_news_model.predict(title_vector, verbose=0)
    

    # On récupère la probabilité sous forme de float
    real_probability_score = float(np.asarray(raw_prediction).ravel()[0])


    # règle de décision : si le score est >= 0.5 alors REAL, sinon FAKE
    predicted_label = "REAL" if real_probability_score >= 0.5 else "FAKE"

    # La confiance correspond au score de la classe prédite
    prediction_confidence = (
        real_probability_score if predicted_label == "REAL"
        else 1 - real_probability_score
    )

    return PredictResponse(
        title=title,
        label=predicted_label,
        confidence=round(prediction_confidence, 2)
    )



# Endpoint de santé

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Vérifier l'état de l'API",
    status_code=status.HTTP_200_OK,
)
async def health() -> HealthResponse:
    """
    Retourne l'état de l'API.
    """
    return HealthResponse(status="ok", model="fake_news_detector")


# Endpoint de prédiction unitaire

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Prédire un titre",
    status_code=status.HTTP_200_OK,
)
async def predict(payload: PredictRequest) -> PredictResponse:
    """
    Reçoit un seul titre et retourne :
    - le titre reçu
    - le label prédit
    - le score de confiance
    """
    validate_title(payload.title)
    return predict_title(payload.title)



# Endpoint de prédiction par lot

@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["Prediction"],
    summary="Prédire une liste de titres",
    status_code=status.HTTP_200_OK,
)
async def predict_titles(payload: BatchRequest) -> BatchResponse:

    if len(payload.titles) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La liste de titres ne peut pas être vide."
        )

    if len(payload.titles) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La liste de titres ne peut pas dépasser 50 éléments."
        )

    batch_predictions = []

    for current_title in payload.titles:
        validate_title(current_title)
        single_prediction = predict_title(current_title)
        batch_predictions.append(single_prediction)

    return BatchResponse(predictions=batch_predictions)