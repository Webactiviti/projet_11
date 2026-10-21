#!/usr/bin/env python
# coding: utf-8

import os
import sys
import getpass
import pandas as pd
import faiss
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv


# ===============================================
#  Chargement des données CSV
# ===============================================
def charger_donnees(fichier_path):
    """Charge le fichier CSV des événements avec gestion des erreurs."""
    try:
        if not Path(fichier_path).is_file():
            raise FileNotFoundError(f"Le fichier '{fichier_path}' est introuvable.")
        
        df = pd.read_csv(fichier_path)
        print(f"{len(df)} lignes chargées depuis '{fichier_path}'.")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV : {e}")
        sys.exit(1)


# ===============================================
#  Indexation FAISS et Embeddings
# ===============================================
def initialiser_index_faiss(df):
    """Initialise le modèle d'embedding et crée l'index FAISS HNSW."""
    try:
        print("--- Chargement et Indexation FAISS ---")
        
        documents = []
        for _, row in df.iterrows():
            texte_vecteur = str(row['texte_a_vectoriser'])
            # Fonction locale pour nettoyer les valeurs NaN ou nulles d'une ligne Pandas
            def get_clean_val(key, default=''):
                val = row.get(key)
                if pd.isna(val) or str(val).strip().lower() in ["", "nan", "none"]:
                    return default
                return str(val).strip()

            metadata = {
                "id": get_clean_val('id', 'N/A'),
                "titre": get_clean_val('titre', 'Titre non disponible'),
                "lieu": get_clean_val('lieu', 'Lieu non spécifié'),
                "ville": get_clean_val('ville', ''),
                "url": get_clean_val('url', ''),
                "date": get_clean_val('date', '')
            }
            documents.append(Document(page_content=texte_vecteur, metadata=metadata))

        # Initialisation du modèle d'embedding
        embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Configuration de l'index HNSW
        dimension = 384
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 64
        index.hnsw.efSearch = 32

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)

        # Création du docstore
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        db_faiss = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        db_faiss.add_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            metadatas=metadatas
        )

        # Sauvegarde locale optionnelle
        db_faiss.save_local("./data/index_openagenda_hnsw")
        print("Index FAISS optimisé (HNSW) créé et sauvegardé avec succès !\n")
        return db_faiss

    except Exception as e:
        print(f"Erreur lors de la création de l'index FAISS ou des embeddings : {e}")
        raise


# ===============================================
#  Configuration du Chatbot Mistral
# ===============================================
def initialiser_chatbot(db_faiss):
    """Configure la chaîne RAG avec le LLM Mistral."""
    try:
        print("--- Configuration du Chatbot Mistral ---")

        load_dotenv()
        
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("La clé API MISTRAL_API_KEY n'est pas définie dans l'environnement.")

        llm = ChatMistralAI(
            model="mistral-tiny",
            temperature=0.3
        )

        retriever = db_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        print("Chatbot Mistral prêt à formuler des recommandations personnalisées !\n")
        return qa_chain

    except Exception as e:
        print(f"Erreur lors de l'initialisation du chatbot Mistral : {e}")
        raise


# ===============================================
#  Exécution d'une requête RAG
# ===============================================
def poser_question(qa_chain, question_utilisateur):
    """Interroge la chaîne RAG et gère les erreurs d'exécution (API, etc.)."""
    try:
        print(f"Question posée : « {question_utilisateur} »\n")
        reponse = qa_chain.invoke({"query": question_utilisateur})
        return reponse
    except Exception as e:
        print(f"Erreur lors de l'exécution de la requête RAG : {e}")
        return None


# ===============================================
# Bloc Principal d'Exécution
# ===============================================
if __name__ == "__main__":
    fichier_csv = "./data/evenement.csv"
    
    # Charger les données du fichier csv
    df_evenements = charger_donnees(fichier_csv)
    
    #  Initialisation de la base vectorielle
    db_vectorstore = initialiser_index_faiss(df_evenements)
    
    #  Initialiser la chaîne de discussion
    chain_rag = initialiser_chatbot(db_vectorstore)
    
    # Exemple d'utilisation
    question = "Je cherche une exposition artistique ou culturelle originale à faire à Paris dans les prochains jours. Que me conseilles-tu ?"
    reponse_rag = poser_question(chain_rag, question)

    # Affichage des réponses et des données METADATA associées aux réponses
    if reponse_rag and "result" in reponse_rag:
        print("=" * 60)
        print(" 🤖  RÉPONSE PERSONNALISÉE DE MISTRAL ")
        print("=" * 60)
        print(reponse_rag["result"].strip())
        print("\n" + "-" * 60)
        print(" 📚  ÉVÉNEMENTS SOURCES UTILISÉS (RAG) ")
        print("-" * 60)



        for i, doc in enumerate(reponse_rag.get("source_documents", []), start=1):
            titre = doc.metadata.get('titre', 'Titre non disponible')
            lieu = doc.metadata.get('lieu', 'Lieu non spécifié')
            ville = doc.metadata.get('ville', '')
            url = doc.metadata.get('url', '')
            
            print(f"[{i}] 📌 {titre}")
            
            # Affichage conditionnel propre du lieu et de la ville
            if ville:
                print(f"    📍 Lieu   : {lieu} ({ville})")
            else:
                print(f"    📍 Lieu   : {lieu}")
                
            if url:
                print(f"    🔗 Lien   : {url}\n")
            else:
                print(f"    🔗 Lien   : Non disponible\n")