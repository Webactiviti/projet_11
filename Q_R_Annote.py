#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
import warnings
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_precision import ContextPrecision

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*uses fixed sampling defaults.*temperature will be ignored.*",
    category=UserWarning,
)

INDEX_PATH = "./data/index_openagenda_hnsw"
CSV_PATH = "./data/evenement.csv"
DATASET_PATH = "./data/annote_dataset.json"

# Jeu de questions / réponses annotées basé sur des événements réels de evenement.csv
DEFAULT_DATASET = [
    {
        "id_test": "q1_adoration",
        "question": "Y a-t-il un temps d'adoration ou de prière à Paris pour entrer dans l'Avent ?",
        "reference": (
            "Oui, l'événement « 24h pour le Seigneur – Adoration » a lieu à "
            "l'Église Sainte-Jeanne de Chantal à Paris les 27 et 28 novembre "
            "(nuit d'adoration pour entrer dans l'Avent et la nouvelle année liturgique). "
            "Lien : https://openagenda.com/sainte-jeanne-de-chantal/events/62525807_24h-pour-le-seigneur-adoration"
        ),
        "expected_event_id": 62525807,
        "categorie": "recherche_thematique",
    },
    {
        "id_test": "q2_benevolat",
        "question": "Je cherche une matinée de bénévolat ou de chantier communautaire le samedi 28 novembre à Paris.",
        "reference": (
            "L'événement « Matinée de bénévolat - samedi 28 novembre 2026 » se déroule "
            "à l'Église Saint-Gervais-Saint-Protais à Paris le Samedi 28 novembre, 09h30, "
            "avec au programme des chantiers communautaires en préparation de l'Avent et un repas partagé. "
            "Lien : https://openagenda.com/jerusalem-paris/events/62892482_journee-des-benevoles-samedi-28-novembre-2026"
        ),
        "expected_event_id": 62892482,
        "categorie": "date_et_activite",
    },
    {
        "id_test": "q3_celebration",
        "question": "Quel événement célèbre Saint Louis et Saint François d'Assise à Paris fin novembre ?",
        "reference": (
            "L'événement « 800 ans après... » a lieu à l'Espace Bernanos à Paris "
            "les 28 et 29 novembre, autour de la célébration de Saint Louis et Saint François d'Assise. "
            "Lien : https://openagenda.com/espace-bernanos/events/7738969_800-ans-apres"
        ),
        "expected_event_id": 7738969,
        "categorie": "recherche_thematique",
    },
    {
        "id_test": "q4_hors_domaine",
        "question": "Quelle est la recette du pain au chocolat ?",
        "reference": (
            "Cette demande est hors du périmètre de la base d'événements culturels et locaux disponible."
        ),
        "expected_event_id": None,
        "categorie": "hors_domaine",
    },
]


def generer_ou_charger_dataset(json_path=DATASET_PATH):
    """Crée le fichier annote_dataset.json si pas présent."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if not os.path.exists(json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_DATASET, f, ensure_ascii=False, indent=2)
        print(f"Fichier annoté générés")
    else:
        print(f"Fichier annoté chargé : {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def initialiser_pipeline_rag():
    """Charge l'index FAISS, LLM ,les wrappers pour RAGAS."""
    load_dotenv(override=True)
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("La clé API GOOGLE_API_KEY n'est pas définie dans l'environnement.")

    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db_faiss = FAISS.load_local(
        INDEX_PATH,
        embeddings_model,
        allow_dangerous_deserialization=True,
    )

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

    retriever = db_faiss.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings_model)

    return db_faiss, qa_chain, ragas_llm, ragas_embeddings


def evaluer_avec_annotations():
  
    annote_dataset = generer_ou_charger_dataset()
    db_faiss, qa_chain, ragas_llm, ragas_embeddings = initialiser_pipeline_rag()

    resultats_detail = []
    ragas_rows = []

    for item in annote_dataset:
        question = item["question"].strip()
        reference = item["reference"].strip()
        expected_id = item.get("expected_event_id")

        #  Recherche vectorielle 
        t_retr_start = time.perf_counter()
        docs_avec_scores = db_faiss.similarity_search_with_score(question, k=3)
        temps_retrieval = time.perf_counter() - t_retr_start

        contexts = [doc.page_content for doc, _ in docs_avec_scores]
        retrieved_ids = [doc.metadata.get("id") for doc, _ in docs_avec_scores]
        scores_faiss = [round(float(score), 3) for _, score in docs_avec_scores]

        if expected_id is None:
            hit_top3 = "N/A (hors domaine)"
        else:
            hit_top3 = expected_id in retrieved_ids

        # Génération de la réponse avec Gemini
        t_gen_start = time.perf_counter()
        reponse = qa_chain.invoke({"query": question})
        temps_generation = time.perf_counter() - t_gen_start

        reponse_texte = (reponse.get("result") or "").strip() if isinstance(reponse, dict) else ""

        resultats_detail.append({
            "id_test": item.get("id_test", ""),
            "categorie": item.get("categorie", ""),
            "expected_id": expected_id,
            "retrieved_ids": retrieved_ids,
            "hit_top3": hit_top3,
            "scores_faiss": scores_faiss,
            "temps_total_s": round(temps_retrieval + temps_generation, 3),
        })

        ragas_rows.append({
            "question": question,
            "answer": reponse_texte if reponse_texte else "Aucune réponse générée.",
            "contexts": contexts if contexts else [""],
            "reference": reference,
        })

    df_detail = pd.DataFrame(resultats_detail)
    print("\n=== Résultats sur le jeu Questions / Réponses annotées ===")
    print(df_detail.to_string(index=False))

    # 3. Évaluation RAGAS complète grâce à la colonne 'reference'
    dataset = Dataset.from_list(ragas_rows)
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
        ContextPrecision(llm=ragas_llm),
    ]

    scores_ragas = evaluate(dataset=dataset, metrics=metrics)
    print("\n=== Scores RAGAS  ===")
    print(scores_ragas)


if __name__ == "__main__":
    evaluer_avec_annotations()
