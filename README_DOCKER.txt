Utilisation :

1. Construire et lancer les services
docker compose up --build

2. Accéder à l'API
http://localhost:8000/docs

3. Accéder à Jupyter
http://localhost:8888

Services lancés :
- api : lance FastAPI avec uvicorn
- jupyter : lance Jupyter Notebook

Le Dockerfile :
- crée l'environnement Python
- installe les dépendances de requirements.txt
- installe jupyter et uvicorn
- copie le projet dans /app
