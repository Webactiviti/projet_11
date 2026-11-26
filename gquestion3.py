#!/usr/bin/env python
# coding: utf-8

from utils_rag_google import executer_requete_rag

if __name__ == "__main__":
    # Question spécifique pour ce script de démonstration
    question = "Quels patrimoines ou visites insolites  on peut visiter ?"
    
    # Appel de la fonction mutualisée
    executer_requete_rag(question)
