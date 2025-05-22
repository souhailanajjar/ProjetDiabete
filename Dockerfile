FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Mise à jour et installation de git + curl pour le healthcheck
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Clonage du repo GitHub (force mise à jour via ARG pour Docker cache)
ARG CACHEBUST=1
RUN git clone https://github.com/souhailanajjar/ProjetDiabete

# Se placer dans le projet
WORKDIR /app/ProjetDiabete

# Mise à jour de pip + installation des dépendances avec timeout
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --retries=10 -r requirements.txt && \
    pip install streamlit

# Exposer le port Streamlit
EXPOSE 8501

# Vérification de santé
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Démarrage de l'app Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
