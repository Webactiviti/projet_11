#!/usr/bin/env python
# coding: utf-8

from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
import pandas as pd

# Charger les variables du fichier .env
load_dotenv()

# Récupération de la clef OpenAgenda
API_KEY = os.getenv("OPENAGENDA_API_KEY")
if not API_KEY:
    raise ValueError("La clé API OpenAgenda n'est pas définie dans l'environnement.")

# URL du site openAgenda .La clé API est placée uniquement dans les en-têtes HTTP (jamais dans l'URL)
BASE_URL = "https://api.openagenda.com/v2"
HEADERS = {"Accept": "application/json", "key": API_KEY}

# Recherche sur Paris
def fetch_agendas(search_query="Paris"):
    """Récupère la liste des agendas basés sur un critère de recherche."""
    url = f"{BASE_URL}/agendas"
    params = {"search": search_query}
    
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        print(f"Erreur lors de la récupération des agendas : {response.status_code}")
        return []
    
    return response.json().get("agendas", [])


def fetch_events_for_agendas(agendas):
    """Parcourt les agendas pour récupérer les événements sur une période donnée."""
    now = datetime.now()
    start_date = now - timedelta(days=182)
    end_date = now + timedelta(days=182) # Ajustez selon vos besoins (ex: ~1 an au total)

    all_events = []
    for agenda in agendas:
        agenda_uid = agenda["uid"]
        events_url = f"{BASE_URL}/agendas/{agenda_uid}/events"

        event_params = {
            "timings[gte]": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timings[lte]": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }  
        
        events_res = requests.get(events_url, headers=HEADERS, params=event_params)
        if events_res.status_code == 200:
            events = events_res.json().get("events", [])
            all_events.extend(events)
            
    return all_events


def clean_and_format_events(all_events):
    """Normalise les données, filtre les colonnes utiles et prépare le texte à vectoriser."""
    if not all_events:
        return pd.DataFrame()

    df = pd.json_normalize(all_events)

    # Utilisation de .reindex() pour éviter les erreurs KeyError si une colonne manque
    colonnes_utiles = [
        'uid', 'title.fr', 'description.fr', 'dateRange.fr', 
        'firstTiming.begin', 'firstTiming.end', 'location.name', 
        'location.address', 'location.city', 'location.latitude', 
        'location.longitude', 'canonicalUrl'
    ]
    
    df = df.reindex(columns=colonnes_utiles)
    
    # Renommer les colonnes
    df.columns = [
        'id', 'titre', 'description', 'date', 'heure debut', 
        'heure fin', 'lieu', 'adresse', 'ville', 'latitude', 'longitude', 'url'
    ]

    # Remplacement des NaN par des chaînes vides
    df_clean = df.fillna("")

    # Création du texte à vectoriser de manière propre
    df_clean['texte_a_vectoriser'] = (
        "Événement : " + df_clean['titre'] + ". " +
        "Description : " + df_clean['description'] + " " +
        "Lieu : " + df_clean['lieu'] + " (" + df_clean['adresse'] + ", " + df_clean['ville'] + "). " +
        "Période : " + df_clean['date']
    )

    return df_clean


if __name__ == "__main__":
    print("Recherche des agendas...")
    agendas = fetch_agendas("Paris")
    print(f"{len(agendas)} agendas trouvés.")

    print("Récupération des événements...")
    all_events = fetch_events_for_agendas(agendas)
    print(f"Nombre total d'événements trouvés : {len(all_events)}")

    print("Nettoyage et formatage des données...")
    df_clean = clean_and_format_events(all_events)

    # Assurez-vous que le dossier ./data existe avant d'enregistrer
    os.makedirs("./data", exist_ok=True)
    df_clean.to_csv("./data/evenement.csv", index=False, encoding="utf-8")
    print("Fichier CSV enregistré avec succès dans ./data/evenement.csv.")