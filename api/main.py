from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import re
import joblib
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# --- INITIALISATION DES RESSOURCES ---
# Téléchargement des ressources NLTK nécessaires au fonctionnement de clean_title
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download('punkt_tab')

app = FastAPI(title="Fake News Detector", version="1.0.0")

# Gestion dynamique des chemins (indépendant du répertoire de lancement)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
VECT_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# Chargement unique au démarrage
try:
    vectorizer = joblib.load(VECT_PATH)
    model = load_model(MODEL_PATH)
    print("Succès : Modèle et Vectoriseur chargés.")
except Exception as e:
    print(f"Erreur critique de chargement : {e}")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_title(text: str) -> str:
    """Fonction de nettoyage identique à la phase d'entraînement."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t, pos="v") for t in tokens if t not in stop_words and len(t) >= 2]
    return " ".join(tokens)

# --- MODÈLES DE DONNÉES (VALIDATION) ---
class PredictRequest(BaseModel):
    title: str = Field(..., description="Le titre de l'article à analyser")

class BatchRequest(BaseModel):
    titles: list[str] = Field(..., description="Liste de titres (max 50)")

# --- ENDPOINTS ---

@app.get("/health")
async def health():
    return {"status": "ok", "model": "fake_news_detector"}

@app.post("/predict")
async def predict(data: PredictRequest):
    # Gestion des cas limites : Titre vide ou espaces
    if not data.title.strip():
        raise HTTPException(status_code=422, detail="Le titre ne peut pas être vide ou composé uniquement d'espaces.")
    
    # Gestion des cas limites : Longueur max
    if len(data.title) > 300:
        raise HTTPException(status_code=400, detail="Le titre dépasse la limite autorisée de 300 caractères.")

    try:
        # Pipeline de prédiction
        text_clean = clean_title(data.title)
        vector = vectorizer.transform([text_clean]).toarray()
        
        # Inférence
        score = float(model.predict(vector, verbose=0)[0][0])
        
        # Formatage de la réponse
        label = "REAL" if score >= 0.5 else "FAKE"
        confidence = score if score >= 0.5 else 1 - score

        return {
            "title": data.title,
            "label": label,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction : {str(e)}")

@app.post("/predict/batch")
async def predict_batch(data: BatchRequest):
    # Gestion des cas limites : Liste vide ou > 50
    if not data.titles or len(data.titles) > 50:
        raise HTTPException(status_code=400, detail="La liste doit contenir entre 1 et 50 titres.")

    results = []
    for title in data.titles:
        if not title.strip() or len(title) > 300:
            # On ignore les titres invalides dans un batch ou on pourrait lever une erreur
            continue
            
        text_clean = clean_title(title)
        vector = vectorizer.transform([text_clean]).toarray()
        score = float(model.predict(vector, verbose=0)[0][0])
        
        label = "REAL" if score >= 0.5 else "FAKE"
        confidence = score if score >= 0.5 else 1 - score
        
        results.append({
            "title": title,
            "label": label,
            "confidence": round(confidence, 2)
        })

    return {"predictions": results}