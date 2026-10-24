#!/usr/bin/env python
# coding: utf-8

from utils_rag import executer_requete_rag

if __name__ == "__main__":
    # Question spécifique pour ce script de démonstration
    question = "Je m'intéresse aux grands sujets de société, à la géopolitique ou aux conférences-débats organisées à Paris. Que me proposes-tu ? "
    
    # Appel de la fonction mutualisée
    executer_requete_rag(question)
