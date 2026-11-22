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
    page_title="Optimisation Collecte Déchets - Casablanca",
    page_icon="🚛",
    layout="wide"
)

st.title("🚛 Optimisation de la collecte des déchets")
st.markdown("Cette application calcule l'itinéraire optimal d'un camion. Si le tonnage total dépasse la capacité, le camion fera **plusieurs voyages**.")

# =========================================================
# INITIALISATION DE LA MÉMOIRE
# =========================================================
if "resultats" not in st.session_state:
    st.session_state.resultats = None

# =========================================================
# BARRE LATÉRALE : PARAMÈTRES
# =========================================================
with st.sidebar:
    st.header("⚙️ Paramètres du camion")
    capacite_camion = st.number_input(
        "Capacité maximale (tonnes)",
        min_value=1.0, max_value=50.0, value=14.0, step=0.5
    )
    vitesse_moyenne = st.number_input(
        "Vitesse moyenne (km/h)",
        min_value=10, max_value=80, value=25, step=5
    )

    st.header("📍 Coordonnées du dépôt (Casablanca)")
    depot_lat = st.number_input("Latitude dépôt", value=33.5950, format="%.6f")
    depot_lon = st.number_input("Longitude dépôt", value=-7.6180, format="%.6f")

    st.header("📂 Données des bacs")
    uploaded_file = st.file_uploader("Importer un fichier CSV ou Excel", type=["csv", "xlsx", "xls"])

# =========================================================
# CHARGEMENT DES DONNÉES
# =========================================================
@st.cache_data
def donnees_exemple():
    return pd.DataFrame({
        "nom": ["Maarif", "Ain Diab", "Sidi Moumen", "Hay Hassani", "Sidi Belyout", "Anfa", "Roches Noires", "Mers Sultan", "Derb Sultan", "Oulfa", "Sidi Maarouf", "Californie"],
        "tonnage": [2.5, 1.8, 3.2, 2.1, 1.5, 2.8, 1.2, 2.0, 1.7, 2.4, 3.0, 1.9],
        "lat": [33.5869, 33.5917, 33.6072, 33.5678, 33.5950, 33.6028, 33.6055, 33.5886, 33.5805, 33.5462, 33.5350, 33.5450],
        "lon": [-7.6320, -7.6780, -7.5380, -7.6650, -7.6180, -7.6520, -7.5980, -7.6230, -7.6350, -7.6520, -7.6410, -7.6210]
    })

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data_bacs = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            data_bacs = pd.read_excel(uploaded_file)
            
        colonnes_requises = {"nom", "tonnage", "lat", "lon"}
        if not colonnes_requises.issubset(data_bacs.columns):
            st.error(f"Le fichier doit contenir les colonnes : {colonnes_requises}")
            st.stop()
        st.success(f"✅ {len(data_bacs)} bacs importés.")
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.stop()
else:
    data_bacs = donnees_exemple()
    st.info("ℹ️ Données d'exemple (Casablanca) utilisées.")

# =========================================================
# AFFICHAGE DES DONNÉES INITIALES
# =========================================================
with st.expander("📋 Voir la liste des bacs", expanded=False):
    st.dataframe(data_bacs, use_container_width=True)
    total_tonnage = data_bacs["tonnage"].sum()
    st.metric("Tonnage total à collecter", f"{total_tonnage:.2f} T")
    nb_voyages_min = int(np.ceil(total_tonnage / capacite_camion))
    st.metric("Nombre minimal de voyages requis", nb_voyages_min)

# =========================================================
# FONCTIONS (DISTANCE ET SOLVEUR MULTI-VOYAGES)
# =========================================================
def calculer_matrice_distances(lats, lons):
    R = 6371000
    n = len(lats)
    matrice = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            lat1, lon1 = np.radians(lats[i]), np.radians(lons[i])
            lat2, lon2 = np.radians(lats[j]), np.radians(lons[j])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            matrice[i][j] = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))
    return matrice.astype(int)

def resoudre_cvrp_multi(distance_matrix, demandes_kg, capacite_kg, nb_voyages):
    n = len(distance_matrix)
    # On indique au solveur qu'il a "nb_voyages" véhicules disponibles (1 camion = X voyages)
    manager = pywrapcp.RoutingIndexManager(n, nb_voyages, 0) 
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demande_callback(from_index):
        return demandes_kg[manager.IndexToNode(from_index)]

    demande_idx = routing.RegisterUnaryTransitCallback(demande_callback)

    # On crée un tableau de capacités pour chaque voyage
    capacites = [capacite_kg] * nb_voyages
    routing.AddDimensionWithVehicleCapacity(demande_idx, 0, capacites, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 10

    return routing.SolveWithParameters(params), manager, routing

# =========================================================
# CALCUL DE L'ITINÉRAIRE
# =========================================================
if st.button("🚀 Calculer l'itinéraire optimal", type="primary", use_container_width=True):
    
    df_complet = pd.concat([
        pd.DataFrame({"nom": ["🏭 Dépôt"], "tonnage": [0.0], "lat": [depot_lat], "lon": [depot_lon]}),
        data_bacs.reset_index(drop=True)
    ], ignore_index=True)

    lats = df_complet["lat"].tolist()
    lons = df_complet["lon"].tolist()
    demandes_kg = [int(t * 1000) for t in df_complet["tonnage"].tolist()]
    capacite_kg = int(capacite_camion * 1000)
    
    # Calcul du nombre de voyages nécessaires
    tonnage_total_a_collecter = sum(demandes_kg)
    nb_voyages = int(np.ceil(tonnage_total_a_collecter / capacite_kg))

    with st.spinner(f"Calcul de la meilleure tournée en cours ({nb_voyages} voyages)..."):
        dist_matrix = calculer_matrice_distances(lats, lons)
        solution, manager, routing = resoudre_cvrp_multi(dist_matrix, demandes_kg, capacite_kg, nb_voyages)

    if solution is None:
        st.error("❌ Erreur mathématique : impossible de trouver une solution.")
        st.session_state.resultats = None
    else:
        # Extraction des tournées
        tournees = []
        distance_totale = 0
        tonnage_total = 0

        for v in range(nb_voyages):
            index = routing.Start(v)
            route = []
            charge = 0
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                tonnage = df_complet.iloc[node]["tonnage"]
                charge += tonnage
                if node != 0: 
                    tonnage_total += tonnage
                
                route.append({
                    "Voyage": v + 1,
                    "Lieu": df_complet.iloc[node]["nom"],
                    "Tonnage (T)": round(tonnage, 2),
                    "Charge cumulée (T)": round(charge, 2),
                    "Lat": df_complet.iloc[node]["lat"],
                    "Lon": df_complet.iloc[node]["lon"]
                })
                
                next_index = solution.Value(routing.NextVar(index))
                distance_totale += routing.GetArcCostForVehicle(index, next_index, v)
                index = next_index
                
            # Retour au dépôt
            route.append({
                "Voyage": v + 1,
                "Lieu": df_complet.iloc[0]["nom"],
                "Tonnage (T)": 0.0,
                "Charge cumulée (T)": round(charge, 2),
                "Lat": df_complet.iloc[0]["lat"],
                "Lon": df_complet.iloc[0]["lon"]
            })
            
            # On ne garde que si le voyage contient au moins un bac (sinon c'est un trajet vide)
            if len(route) > 2:
                tournees.extend(route)

        st.session_state.resultats = {
            "tournees": tournees,
            "distance_totale": distance_totale,
            "tonnage_total": tonnage_total,
            "capacite_camion": capacite_camion,
            "vitesse_moyenne": vitesse_moyenne,
            "lats": lats,
            "lons": lons,
            "nb_voyages": nb_voyages
        }

# =========================================================
# AFFICHAGE DES RÉSULTATS
# =========================================================
if st.session_state.resultats is not None:
    res = st.session_state.resultats
    
    st.header(f"📊 Résultats de l'optimisation ({res['nb_voyages']} voyage(s))")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Distance totale", f"{res['distance_totale']/1000:.2f} km")
    col2.metric("Tonnage collecté", f"{res['tonnage_total']:.2f} T")
    col3.metric("Capacité camion", f"{res['capacite_camion']:.2f} T")
    duree_h = (res['distance_totale'] / 1000) / res['vitesse_moyenne']
    col4.metric("Durée estimée", f"{duree_h:.2f} h")

    st.subheader("🗺️ Ordre de collecte par voyage")
    df_route = pd.DataFrame(res['tournees'])
    st.dataframe(df_route, use_container_width=True)

    st.subheader("📍 Carte de l'itinéraire")
    m = folium.Map(location=[np.mean(res['lats']), np.mean(res['lons'])], zoom_start=12)
    
    # Couleurs pour différencier les voyages
    couleurs = ["blue", "purple", "orange", "darkred", "cadetblue"]

    for v in range(1, res['nb_voyages'] + 1):
        df_voyage = df_route[df_route["Voyage"] == v]
        if df_voyage.empty: continue
        
        coords = []
        for i, row in df_voyage.iterrows():
            icone = "home" if "Dépôt" in row["Lieu"] else "trash"
            couleur_marqueur = "red" if icone == "home" else "green"
            
            folium.Marker(
                location=[row["Lat"], row["Lon"]],
                popup=f"Voyage {v} - {row['Lieu']} ({row['Tonnage (T)']} T)",
                icon=folium.Icon(color=couleur_marqueur, icon=icone, prefix="fa")
            ).add_to(m)
            coords.append([row["Lat"], row["Lon"]])
            
        # Tracé de la ligne pour ce voyage spécifique
        folium.PolyLine(coords, color=couleurs[(v-1) % len(couleurs)], weight=4, opacity=0.8, tooltip=f"Voyage {v}").add_to(m)

    st_folium(m, width=None, height=500, use_container_width=True)

    csv_export = df_route.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger l'itinéraire complet (CSV)", csv_export, "itineraires_multi_voyages.csv", "text/csv")
