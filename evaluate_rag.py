#!/usr/bin/env python
# coding: utf-8

import os
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

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

import utils_rag_google

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*uses fixed sampling defaults.*temperature will be ignored.*",
    category=UserWarning,
)

INDEX_PATH = "./data/index_openagenda_hnsw"

QUESTIONS_TEST = [
    "quelle la recette du pain au chocolat ?",
    "Je cherche une exposition artistique ou culturelle originale à faire à Paris dans les prochains jours.",
]


def initialiser_pipeline_evaluation():
    """Charge l'index FAISS, la chaîne RAG et les wrappers pour RAGAS."""
    load_dotenv(override=True)
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("La clé API GOOGLE_API_KEY n'est pas définie dans l'environnement.")

    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db_faiss = FAISS.load_local(
        INDEX_PATH,
        embeddings_model,
        allow_dangerous_deserialization=True,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
    )

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


def main():
    db_faiss, qa_chain, ragas_llm, ragas_embeddings = initialiser_pipeline_evaluation()

    perf_logs = []
    ragas_rows = []

    for question in QUESTIONS_TEST:
        q = question.strip()

        t_retr_start = time.perf_counter()
        docs_avec_scores = db_faiss.similarity_search_with_score(q, k=3)
        temps_retrieval = time.perf_counter() - t_retr_start

        contexts = [doc.page_content for doc, _score in docs_avec_scores]
        scores_faiss = [round(float(score), 3) for _doc, score in docs_avec_scores]

        t_gen_start = time.perf_counter()
        reponse = qa_chain.invoke({"query": q})
        temps_generation = time.perf_counter() - t_gen_start

        reponse_texte = (reponse.get("result") or "").strip() if isinstance(reponse, dict) else ""

        perf_logs.append({
            "question": q,
            "temps_retrieval_s": round(temps_retrieval, 3),
            "temps_generation_s": round(temps_generation, 3),
            "temps_total_s": round(temps_retrieval + temps_generation, 3),
            "scores_faiss": scores_faiss,
            "nb_sources": len(contexts),
        })

        ragas_rows.append({
            "question": q,
            "answer": reponse_texte if reponse_texte else "Aucune réponse générée.",
            "contexts": contexts if contexts else [""],
        })

    df_perf = pd.DataFrame(perf_logs)
    print("=== Métriques techniques & FAISS ===")
    print(df_perf.to_string(index=False))

    dataset = Dataset.from_list(ragas_rows)
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
    ]

    scores_ragas = evaluate(
        dataset=dataset,
        metrics=metrics,
    )
    print("\n=== Métriques RAGAS ===")
    print(scores_ragas)


if __name__ == "__main__":
    assert hasattr(utils_rag_google, "executer_requete_rag")
    main()