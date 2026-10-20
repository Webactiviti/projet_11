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

Déroulement du programme

graph TD
    subgraph P1 ["Programme 1 : openagenda_v2.py (Collecte & Préparation)"]
        A[Chargement .env] --> B[Requête API OpenAgenda]
        B --> C[Récupération des Agendas et Événements]
        C --> D[Normalisation et Nettoyage des données]
        D --> E[Création du texte à vectoriser]
        E --> F[Exportation : evenement.csv]
    end

    subgraph P2 ["Programme 2 : app_rag_v2.py (Indexation & RAG)"]
        F --> G[Chargement du fichier CSV]
        G --> H[Création des Documents LangChain et Métadonnées]
        H --> I[Génération des Embeddings Textuels]
        I --> J[Indexation vectorielle HNSW avec FAISS]
        J --> K[Configuration du LLM Mistral et Chaîne QA]
        K --> L[Requête / Question Utilisateur]
        L --> M[Recherche par Similarité RAG]
        M --> N[Génération de la Réponse et des Sources]
    end

    style P1 fill:#f9f,stroke:#333,stroke-width:2px
    style P2 fill:#bbf,stroke:#333,stroke-width:2px



# installation

 1. Copier le projet dans un répertoire de votre choix

 2. Python et uv doivent être déjà installé sur votre machine

 3. Lncer la commande : uv sync , pour importer les librairies du projet

 4. Renommer exemple.env en .env pour mettre les différentes clefs API : Hugging Face , OpenAgenda et Mistral

 5. Lancer la commande uv run jupyter lab : openagenda.ipynb pour lire un agenda et pour le RAG : app_rag.ipynb

## 📂 Structure du Répertoire

```text
projet_11

├── pyproject.toml                                  # Gestion des dépendances Python (uv)
├── data                                            # répertoire  de document 
|    └── evenement.csv                              # fichier evenement sélectioné dans openagenda extrait et préparé pour LLM
|    └── index_openagenda_hnsw                      # Répertoire fichierss : Index FAISS optimisé (HNSW) 
|         └── index.faiss    
|         └── index.pkl  
|    └── dico_data
|         └── dico_data_opa.xls                     # dictionnaire  de données fichiers openagenda
├── openagenda.ipynb                                # notebook gestion data openagenda
├── app_rag.ipynb                                   # notebook gestion RAG et gestion avec Mistral
├── example.env                                     # fichier exemple .env pour les clefs API
└── README.md                                       # Documentation du projet
```
