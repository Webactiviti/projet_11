#!/usr/bin/env python
# coding: utf-8

from utils_rag import executer_requete_rag

if __name__ == "__main__":
    # Question spécifique pour ce script de démonstration
    question = "Quels sont les ateliers culturels, créatifs ou participatifs auxquels je peux m'inscrire ? "
    
    # Appel de la fonction mutualisée
    executer_requete_rag(question)
