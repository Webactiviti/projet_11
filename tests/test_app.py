import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import des fonctions depuis vos scripts respectifs
from app_rag_V2 import charger_donnees
from openagenda_V2 import clean_and_format_events, fetch_agendas


# ==========================================
# 1. Tests pour app_rag_V2.py (Chargement CSV)
# ==========================================

def test_charger_donnees_valide(tmp_path):
    """Teste le chargement d'un fichier CSV valide avec des données propres."""
    d = tmp_path / "data"
    d.mkdir()
    f = d / "evenement.csv"
    
    # Création d'un contenu CSV factice
    contenu_csv = "id,texte_a_vectoriser,titre,lieu,ville,url,date\n1,Texte test,Exposition Art,Musée Louvre,Paris,http://test.com,2026-06-01"
    f.write_text(contenu_csv, encoding="utf-8")
    
    df = charger_donnees(str(f))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.loc[0, 'titre'] == "Exposition Art"


def test_charger_donnees_introuvable():
    """Vérifie que le script lève bien un SystemExit si le fichier CSV est absent."""
    with pytest.raises(SystemExit):
        charger_donnees("./data/fichier_absolument_inexistant.csv")


# ==========================================
# 2. Tests pour openagenda_V2.py (Nettoyage des données)
# ==========================================

def test_clean_and_format_events_vide():
    """Vérifie que le formatage gère correctement une liste d'événements vide."""
    df_result = clean_and_format_events([])
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.empty


def test_clean_and_format_events_nominal():
    """Teste la normalisation et la création du texte à vectoriser à partir de données brutes."""
    # Simulation d'un retour brut de l'API OpenAgenda
    raw_events = [
        {
            "uid": 123,
            "title": {"fr": "Festival de Jazz"},
            "description": {"fr": "Un super festival en plein air."},
            "dateRange": {"fr": "Du 1 au 3 juillet"},
            "firstTiming": {"begin": "2026-07-01T10:00:00Z", "end": "2026-07-01T22:00:00Z"},
            "location": {
                "name": "Parc Floral",
                "address": "Route de la Pyramide",
                "city": "Paris",
                "latitude": 48.83,
                "longitude": 2.44
            },
            "canonicalUrl": "https://openagenda.com/festival-jazz"
        }
    ]

    df_clean = clean_and_format_events(raw_events)

    # Vérifications sur le DataFrame transformé
    assert not df_clean.empty
    assert 'texte_a_vectoriser' in df_clean.columns
    assert df_clean.loc[0, 'titre'] == "Festival de Jazz"
    assert df_clean.loc[0, 'ville'] == "Paris"
    
    # Vérification que le texte à vectoriser est bien construit
    texte_attendu = "Événement : Festival de Jazz. Description : Un super festival en plein air. Lieu : Parc Floral (Route de la Pyramide, Paris). Période : Du 1 au 3 juillet"
    assert df_clean.loc[0, 'texte_a_vectoriser'] == texte_attendu


# ==========================================
# 3. Tests avec Mock (Simulation d'API externe)
# ==========================================

@patch("openagenda_V2.requests.get")
def test_fetch_agendas_succes(mock_get):
    """Teste la récupération des agendas avec un mock de l'appel HTTP requests.get."""
    # Configuration du comportement de la réponse HTTP simulée
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"agendas": [{"uid": 456, "title": "Agenda Paris"}]}
    mock_get.return_value = mock_response

    agendas = fetch_agendas("Paris")
    
    assert len(agendas) == 1
    assert agendas[0]["uid"] == 456
    mock_get.assert_called_once()


@patch("openagenda_V2.requests.get")
def test_fetch_agendas_erreur_http(mock_get):
    """Vérifie le comportement si l'API OpenAgenda renvoie une erreur (ex: 500 ou 404)."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    agendas = fetch_agendas("Paris")
    
    # Doit retourner une liste vide en cas d'erreur de statut
    assert agendas == []
