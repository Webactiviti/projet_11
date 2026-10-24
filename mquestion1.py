#!/usr/bin/env python
# coding: utf-8

from utils_rag import executer_requete_rag

if __name__ == "__main__":
    # Question spécifique pour ce script de démonstration
    question = "Y a-t-il des concerts ou des événements musicaux prévus à Paris ? "
    
    # Appel de la fonction mutualisée
    executer_requete_rag(question)
