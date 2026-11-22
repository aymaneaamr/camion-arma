import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# =========================================================
# CONFIGURATION DE LA PAGE
# =========================================================
st.set_page_config(
    page_title="Optimisation Collecte Déchets",
    page_icon="🚛",
    layout="wide"
)

st.title("🚛 Optimisation de l'itinéraire de collecte des déchets")
st.markdown("Cette application calcule l'itinéraire optimal d'un camion de collecte en fonction de la **capacité maximale du camion** et du **tonnage de chaque bac**.")

# =========================================================
# BARRE LATÉRALE : PARAMÈTRES
# =========================================================
with st.sidebar:
    st.header("⚙️ Paramètres du camion")
    capacite_camion = st.number_input(
        "Capacité maximale du camion (tonnes)",
        min_value=1.0, max_value=50.0, value=10.0, step=0.5
    )
    vitesse_moyenne = st.number_input(
        "Vitesse moyenne (km/h)",
        min_value=10, max_value=80, value=30, step=5
    )

    st.header("📍 Coordonnées du dépôt")
    depot_lat = st.number_input("Latitude dépôt", value=48.8566, format="%.6f")
    depot_lon = st.number_input("Longitude dépôt", value=2.3522, format="%.6f")

    st.header("📂 Données des bacs")
    uploaded_file = st.file_uploader(
        "Importer un fichier CSV (colonnes : nom, tonnage, lat, lon)",
        type="csv"
    )

    st.markdown("---")
    st.markdown("**Exemple de CSV attendu :**")
    st.code("nom,tonnage,lat,lon\nBac A,1.2,48.85,2.35\nBac B,2.5,48.86,2.36", language="text")


# =========================================================
# CHARGEMENT DES DONNÉES
# =========================================================
@st.cache_data
def donnees_exemple():
    return pd.DataFrame({
        "nom": ["Bac A", "Bac B", "Bac C", "Bac D", "Bac E", "Bac F", "Bac G"],
        "tonnage": [1.2, 2.5, 0.8, 3.0, 1.5, 2.2, 1.8],
        "lat": [48.8500, 48.8600, 48.8400, 48.8700, 48.8300, 48.8550, 48.8650],
        "lon": [2.3500, 2.3600, 2.3400, 2.3700, 2.3300, 2.3450, 2.3650]
    })


if uploaded_file is not None:
    try:
        data_bacs = pd.read_csv(uploaded_file)
        colonnes_requises = {"nom", "tonnage", "lat", "lon"}
        if not colonnes_requises.issubset(data_bacs.columns):
            st.error(f"Le CSV doit contenir les colonnes : {colonnes_requises}")
            st.stop()
        st.success(f"✅ {len(data_bacs)} bacs importés depuis le CSV.")
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        st.stop()
else:
    data_bacs = donnees_exemple()
    st.info("ℹ️ Aucun fichier importé — utilisation des données d'exemple.")

# =========================================================
# AFFICHAGE DES DONNÉES
# =========================================================
with st.expander("📋 Voir la liste des bacs", expanded=False):
    st.dataframe(data_bacs, use_container_width=True)
    total_tonnage = data_bacs["tonnage"].sum()
    st.metric("Tonnage total à collecter", f"{total_tonnage:.2f} T")
    nb_voyages_min = int(np.ceil(total_tonnage / capacite_camion))
    st.metric("Nombre minimal de voyages requis", nb_voyages_min)


# =========================================================
# FONCTIONS UTILITAIRES
# =========================================================
def calculer_matrice_distances(lats, lons):
    """Calcule une matrice de distances en mètres (formule de Haversine)."""
    R = 6371000  # rayon Terre en mètres
    n = len(lats)
    matrice = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lat1, lon1 = np.radians(lats[i]), np.radians(lons[i])
            lat2, lon2 = np.radians(lats[j]), np.radians(lons[j])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            matrice[i][j] = R * c
    return matrice.astype(int)


def resoudre_cvrp(distance_matrix, demandes_kg, capacite_kg):
    """Résout le problème du voyageur de commerce avec contraintes de capacité (CVRP)."""
    n = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # 1 véhicule, dépôt = index 0
    routing = pywrapcp.RoutingModel(manager)

    # Callback distances
    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Callback demande (kg)
    def demande_callback(from_index):
        return demandes_kg[manager.IndexToNode(from_index)]

    demande_idx = routing.RegisterUnaryTransitCallback(demande_callback)

    # Contrainte de capacité
    routing.AddDimensionWithVehicleCapacity(
        demande_idx,
        0,
        [capacite_kg],
        True,
        "Capacity"
    )

    # Paramètres de recherche
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 15

    solution = routing.SolveWithParameters(params)
    return solution, manager, routing


# =========================================================
# CALCUL DE L'ITINÉRAIRE
# =========================================================
if st.button("🚀 Calculer l'itinéraire optimal", type="primary", use_container_width=True):

    # Préparation : dépôt (index 0) + bacs
    df_complet = pd.concat([
        pd.DataFrame({"nom": ["🏭 Dépôt"], "tonnage": [0.0], "lat": [depot_lat], "lon": [depot_lon]}),
        data_bacs.reset_index(drop=True)
    ], ignore_index=True)

    lats = df_complet["lat"].tolist()
    lons = df_complet["lon"].tolist()
    demandes_kg = [int(t * 1000) for t in df_complet["tonnage"].tolist()]
    capacite_kg = int(capacite_camion * 1000)

    with st.spinner("Calcul de la meilleure tournée en cours..."):
        dist_matrix = calculer_matrice_distances(lats, lons)

        try:
            solution, manager, routing = resoudre_cvrp(dist_matrix, demandes_kg, capacite_kg)
        except Exception as e:
            st.error(f"Erreur dans le solveur : {e}")
            st.stop()

    if solution is None:
        st.error("❌ Aucune solution trouvée. Vérifiez que la capacité du camion est suffisante.")
        st.stop()

    # ---- Extraction de la tournée ----
    index = routing.Start(0)
    ordre_visite = []
    distance_totale = 0
    charge_cumulee = 0
    tonnage_total = 0

    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        tonnage = df_complet.iloc[node]["tonnage"]
        charge_cumulee += tonnage
        if node != 0:
            tonnage_total += tonnage
        ordre_visite.append({
            "Étape": len(ordre_visite),
            "Lieu": df_complet.iloc[node]["nom"],
            "Tonnage (T)": round(tonnage, 2),
            "Charge cumulée (T)": round(charge_cumulee, 2),
            "Lat": df_complet.iloc[node]["lat"],
            "Lon": df_complet.iloc[node]["lon"]
        })
        next_index = solution.Value(routing.NextVar(index))
        distance_totale += routing.GetArcCostForVehicle(index, next_index, 0)
        index = next_index

    # Retour au dépôt
    ordre_visite.append({
        "Étape": len(ordre_visite),
        "Lieu": df_complet.iloc[0]["nom"],
        "Tonnage (T)": 0.0,
        "Charge cumulée (T)": round(charge_cumulee, 2),
        "Lat": df_complet.iloc[0]["lat"],
        "Lon": df_complet.iloc[0]["lon"]
    })

    # ---- Résumé ----
    st.header("📊 Résultats de l'optimisation")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Distance totale", f"{distance_totale/1000:.2f} km")
    col2.metric("Tonnage collecté", f"{tonnage_total:.2f} T")
    col3.metric("Capacité camion", f"{capacite_camion:.2f} T")
    duree_h = (distance_totale / 1000) / vitesse_moyenne
    col4.metric("Durée estimée", f"{duree_h:.2f} h")

    taux_remplissage = (tonnage_total / capacite_camion) * 100
    st.progress(min(taux_remplissage / 100, 1.0), text=f"Taux de remplissage : {taux_remplissage:.1f}%")

    # ---- Tableau de la tournée ----
    st.subheader("🗺️ Ordre de visite")
    df_route = pd.DataFrame(ordre_visite)
    st.dataframe(df_route, use_container_width=True)

    # ---- Carte interactive ----
    st.subheader("📍 Carte de l'itinéraire")
    centre_lat = np.mean(lats)
    centre_lon = np.mean(lons)
    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=13, tiles="OpenStreetMap")

    # Marqueurs
    for i, row in df_route.iterrows():
        if "Dépôt" in row["Lieu"]:
            couleur, icone = "red", "home"
        else:
            couleur, icone = "green", "trash"
        folium.Marker(
            location=[row["Lat"], row["Lon"]],
            popup=f"{row['Lieu']} — {row['Tonnage (T)']} T",
            tooltip=f"Étape {row['Étape']} : {row['Lieu']}",
            icon=folium.Icon(color=couleur, icon=icone, prefix="fa")
        ).add_to(m)

    # Tracé de l'itinéraire
    coords = [[r["Lat"], r["Lon"]] for r in ordre_visite]
    folium.PolyLine(coords, color="blue", weight=4, opacity=0.8).add_to(m)

    st_folium(m, width=None, height=500, use_container_width=True)

    # ---- Téléchargement ----
    csv_export = df_route[["Étape", "Lieu", "Tonnage (T)", "Charge cumulée (T)", "Lat", "Lon"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger l'itinéraire (CSV)",
        csv_export,
        "itineraire_collecte.csv",
        "text/csv"
    )
