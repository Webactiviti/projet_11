#!/usr/bin/env python
# coding: utf-8

from utils_rag import executer_requete_rag

if __name__ == "__main__":
    # Question spécifique pour ce script de démonstration
    question = "Je m'intéresse au cinéma. Quelles sont les avant-premières ou les projections de films proposées ? "
    
    # Appel de la fonction mutualisée
    executer_requete_rag(question)
