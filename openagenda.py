#!/usr/bin/env python
# coding: utf-8

# import des librairies

from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
import pandas as pd


# Charger les variables du fichier .env Pour la gestion des clefs API
load_dotenv()

# Récupération de la clef OpenAgenda
API_KEY = os.getenv("OPENAGENDA_API_KEY")
if not API_KEY:
    raise ValueError(
        "La clé API OpenAgenda n'est pas définie dans l'environnement."
    )
# URL du site openAgenda
BASE_URL = "https://api.openagenda.com/v2"

# La clé API est placée uniquement dans les en-têtes HTTP (jamais dans l'URL)
headers = {"Accept": "application/json", "key": API_KEY}

# Rechercher les agendas basés à Paris
agendas_url = f"{BASE_URL}/agendas"
params = {"search": "Paris"} 

#interrogation du site openAgenda
response = requests.get(agendas_url, headers=headers, params=params)

#Vérification du code retour de la requête
if response.status_code != 200:
    print(f"Erreur lors de la récupération des agendas : {response.status_code}")
    exit()

# Récupération de la liste des agendas
agendas = response.json().get("agendas", [])

# Normalisation Json en data Frame
df_ag = pd.json_normalize(agendas)



# Définition de la période (sur +/- - 6 mois ) 
now = datetime.now()- timedelta(days=128)
one_year_later = now + timedelta(days=128)

# Parcourir les agendas pour récupérer les événements avec la contrainte temporelle
all_events = []
for agenda in agendas:
    agenda_uid = agenda["uid"]
    events_url = f"{BASE_URL}/agendas/{agenda_uid}/events"

    event_params = {
        "timings[gte]": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timings[lte]": one_year_later.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }  
    #Requete sur l'agenda avec paramètre uid
    events_res = requests.get(events_url, headers=headers, params=event_params)
    # Réponse du site
    if events_res.status_code == 200:
        events = events_res.json().get("events", [])
        all_events.extend(events)

print( f"Nombre total d'événements trouvés sur Paris via l'API v2 : {len(all_events)}")

# Aplatissement des données imbriquées
df = pd.json_normalize(all_events)


# Gestion des colonnes utiles pour notre POC

colonnes_utiles_completes = [
    'uid', 
    'title.fr', 
    'description.fr',        # description 
    'dateRange.fr',          # Date lisible en français
    'firstTiming.begin',     # Date/heure de début (pour les tris)
    'firstTiming.end',       # Date/heure de fin
    'location.name',         # Nom du lieu
    'location.address',      # Adresse complète
    'location.city',         # Ville
    'location.latitude',     # Latitude (carte)
    'location.longitude',    # Longitude (carte)
    'canonicalUrl'           # Lien vers la page de l'événement
]

df_selection = df[colonnes_utiles_completes].copy()
# Renommer pour plus de simplicité si besoin
df_selection.columns = ['id', 'titre', 'description','date' ,'heure debut', 'heure fin','lieu', 'adresse', 'ville', 'latitude','longitude','url']

pd.set_option('display.max_columns',None)
df_selection.head()


# Clean df_selection et  texte
# Remplace NaN par chaines vides
df_clean = df_selection.fillna("")


# Combiner les colonnes en un texte descriptif  pour chaque événement
df_clean['texte_a_vectoriser'] = (
    "Événement : " + df_clean['titre'] + ". " +
    "Description : " + df_clean['description'] + " " +
    "Lieu : " + df_clean['lieu'] + " (" + df_clean['adresse'] + ", " + df_clean['ville'] + "). " +
    "Période : " + df_clean['date']
)


# Enregistrement du data frame dans un fichier CSV

df_clean.to_csv("./data/evenement.csv", index = False , encoding ="utf-8")



