#!/usr/bin/env python
# coding: utf-8

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_classic.chains import RetrievalQA

def executer_requete_rag(question_utilisateur):

    os.system('clear')
    """Charge l'index FAISS, configure le chatbot Mistral et pose la question."""
    load_dotenv()
    
    # Vérification de la clé API Mistral
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("La clé API MISTRAL_API_KEY n'est pas définie dans l'environnement.")

    # Chargement du modèle d'embedding
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Chargement de la base vectorielle sauvegardée
    db_faiss = FAISS.load_local(
        "./data/index_openagenda_hnsw", 
        embeddings_model, 
        allow_dangerous_deserialization=True
    )

    # Configuration du Chatbot Mistral
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
    
    # Exécution de la requête
    print(f"\n💬 Question posée : « {question_utilisateur} »\n")
    reponse = qa_chain.invoke({"query": question_utilisateur})

    # Affichage formaté de la réponse et des sources
    if reponse and "result" in reponse:
        print("=" * 60)
        print(" 🤖  RÉPONSE PERSONNALISÉE DE MISTRAL ")
        print("=" * 60)
        print(reponse["result"].strip())
        print("\n" + "-" * 60)
        print(" 📚  ÉVÉNEMENTS SOURCES UTILISÉS (RAG) ")
        print("-" * 60)

        for i, doc in enumerate(reponse.get("source_documents", []), start=1):
            titre = doc.metadata.get('titre', 'Titre non disponible')
            lieu = doc.metadata.get('lieu', 'Lieu non spécifié')
            ville = doc.metadata.get('ville', '')
            url = doc.metadata.get('url', '')
            
            print(f"[{i}] 📌 {titre}")
            if ville:
                print(f"    📍 Lieu   : {lieu} ({ville})")
            else:
                print(f"    📍 Lieu   : {lieu}")
                
            if url:
                print(f"    🔗 Lien   : {url}\n")
            else:
                print(f"    🔗 Lien   : Non disponible\n")
