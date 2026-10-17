# Concevez et deployer un système RAG

# Système RAG en Chatbot pour les événements sur Paris (  0penagenda + FAISS (Hugging face) + Mistral)


# Description 

Mise en place d'un pipeline

1. Récupération des données sélectionées sur Openagenda

2. Mise en forme des data et préparation et enregistrement dans un fichier CSV

3. Lecture du fichier , LangChain , Indexation FAISS & Maintien des ID ( pour info supplémentaire dans metadata )


Pour le projet :

Clef API à se procurer
 
    - Openagenda
    - Hugging Face
    - Mistral AI


# installation

 1. Copier le projet dans un répertoire de votre choix

 2. Python et uv doivent être déjà installé sur votre machine

 3. Lncer la commande : uv sync , pour importer les librairies du projet

 4. Renommer exemple.env en .env pour mettre les différentes clefs API : Hugging Face , OpenAgenda et Mistral

 5. Lancer la commande uv run jupyter lab : openagenda.ipynb pour lire un agenda et pour le RAG : app_rag.ipynb

## 📂 Structure du Répertoire

```text
projet_11/

├── pyproject.toml                                  # Gestion des dépendances Python (uv)
├── data                                            # répertoire  de document 
|    └── evenement.csv                              # fichier evenement sélectioné dans openagenda extrait et préparé pour LLM
|    └── index_openagenda_hnsw                      # Répertoire fichierss : Index FAISS optimisé (HNSW) 
|         └── index.faiss    
|         └── index.pkl  
|    └── dico_data
|         └── dico_data_opa                         # dictionnaire  de données fichiers openagenda
├── openagenda.ipynb                                # notebook gestion data openagenda
├── app_rag.ipynb                                   # notebook gestion RAG et gestion avec Mistral
├── example.env                                     # fichier exemple .env pour les clefs API
└── README.md                                       # Documentation du projet
```
