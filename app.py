
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.collections import LineCollection
import random
import math

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ACO - Recherche de nourriture",
    page_icon="🐜",
    layout="wide"
)

ROWS = 14
COLS = 20

NEST = (12, 1)
FOOD = (1, 18)

OBSTACLES = {
    (3, 4), (3, 5), (3, 6),
    (4, 6), (5, 6),
    (7, 3), (7, 4), (7, 5),
    (8, 12), (8, 13), (8, 14),
    (9, 14), (10, 14),
    (3, 15), (4, 15), (5, 15),
    (11, 8), (11, 9)
}

DIRECTIONS = [
    (-1, 0), (1, 0),
    (0, -1), (0, 1),
    (-1, -1), (-1, 1),
    (1, -1), (1, 1)
]

# ============================================================
# SESSION STATE
# ============================================================

def initialize():
    st.session_state.pheromones = np.ones((ROWS, COLS), dtype=float)
    st.session_state.ants = []
    st.session_state.iteration = 1
    st.session_state.step = 0
    st.session_state.best_path = None
    st.session_state.best_length = float("inf")
    st.session_state.completed_paths = []
    st.session_state.history = []
    st.session_state.current_ant = 0
    st.session_state.stage = "initialisation"
    st.session_state.last_candidates = []
    st.session_state.last_decision = None
    st.session_state.message = "Simulation initialisée."
    st.session_state.started = False

    for i in range(st.session_state.nb_ants):
        st.session_state.ants.append({
            "id": i + 1,
            "position": NEST,
            "path": [NEST],
            "phase": "aller",
            "status": "active",
            "length": 0.0,
            "steps": 0,
            "color": plt.cm.tab20(i % 20),
        })


if "nb_ants" not in st.session_state:
    st.session_state.nb_ants = 5

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# ============================================================
# ENVIRONMENT
# ============================================================

def valid_cell(cell):
    r, c = cell

    return (
        0 <= r < ROWS
        and 0 <= c < COLS
        and cell not in OBSTACLES
    )


def distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def get_neighbors(position):
    r, c = position
    neighbors = []

    for dr, dc in DIRECTIONS:
        candidate = (r + dr, c + dc)

        if not valid_cell(candidate):
            continue

        # Prevent diagonal movement through obstacle corners.
        if dr != 0 and dc != 0:
            if not valid_cell((r + dr, c)):
                continue
            if not valid_cell((r, c + dc)):
                continue

        neighbors.append(candidate)

    return neighbors


# ============================================================
# HEURISTIC AND PROBABILITY
# ============================================================

def calculate_probabilities(ant, alpha, beta, pheromones):
    position = ant["position"]
    candidates = get_neighbors(position)

    if not candidates:
        return []

    data = []
    total = 0.0

    for cell in candidates:
        d = distance(position, cell)

        tau = max(float(pheromones[cell]), 0.001)

        if ant["phase"] == "aller":
            remaining_distance = distance(cell, FOOD)
        else:
            remaining_distance = distance(cell, NEST)

        eta = 1.0 / max(remaining_distance, 0.1)

        # Discourage immediate loops and repeated visits.
        visit_count = ant["path"].count(cell)
        revisit_factor = 0.15 ** visit_count

        weight = (
            (tau ** alpha) *
            (eta ** beta) *
            revisit_factor
        )

        data.append({
            "cell": cell,
            "distance": d,
            "tau": tau,
            "eta": eta,
            "weight": weight
        })

        total += weight

    for item in data:
        item["probability"] = (
            item["weight"] / total if total > 0 else 0
        )

    return data


def choose_next_position(candidates):
    if not candidates:
        return None

    probabilities = [
        item["probability"] for item in candidates
    ]

    index = random.choices(
        range(len(candidates)),
        weights=probabilities,
        k=1
    )[0]

    return candidates[index]["cell"]


# ============================================================
# ANT MOVEMENT
# ============================================================

def move_ant(ant, alpha, beta):
    if ant["status"] != "active":
        return

    candidates = calculate_probabilities(
        ant,
        alpha,
        beta,
        st.session_state.pheromones
    )

    st.session_state.last_candidates = candidates
    st.session_state.current_ant = ant["id"]

    if not candidates:
        ant["status"] = "bloquee"
        st.session_state.message = (
            f"La fourmi {ant['id']} est bloquée."
        )
        return

    next_position = choose_next_position(candidates)

    selected = next(
        item for item in candidates
        if item["cell"] == next_position
    )

    st.session_state.last_decision = selected

    step_distance = distance(
        ant["position"],
        next_position
    )

    ant["length"] += step_distance
    ant["steps"] += 1
    ant["position"] = next_position
    ant["path"].append(next_position)

    # The ant finds the food.
    if next_position == FOOD and ant["phase"] == "aller":
        ant["phase"] = "retour"
        st.session_state.message = (
            f"🐜 Fourmi {ant['id']} a trouvé la nourriture ! "
            "Elle retourne au nid."
        )
        return

    # The ant returns to the nest.
    if next_position == NEST and ant["phase"] == "retour":
        ant["status"] = "terminée"

        route = ant["path"].copy()
        length = ant["length"]

        st.session_state.completed_paths.append({
            "fourmi": ant["id"],
            "chemin": route,
            "distance": length
        })

        if length < st.session_state.best_length:
            st.session_state.best_length = length
            st.session_state.best_path = route

        st.session_state.message = (
            f"Fourmi {ant['id']} est revenue au nid. "
            f"Distance parcourue : {length:.2f}"
        )
        return

    # Safety limit to avoid endless wandering.
    if ant["steps"] >= ROWS * COLS * 2:
        ant["status"] = "bloquee"
        st.session_state.message = (
            f"La fourmi {ant['id']} a atteint la limite de déplacement."
        )
        return

    st.session_state.message = (
        f"Fourmi {ant['id']} avance vers {next_position}."
    )


# ============================================================
# PHEROMONE UPDATE
# ============================================================

def update_pheromones(rho, q):
    pheromones = st.session_state.pheromones

    # Evaporation.
    pheromones *= (1 - rho)

    # Reinforcement by successful ants.
    for result in st.session_state.completed_paths:
        route = result["chemin"]
        length = result["distance"]

        if length <= 0:
            continue

        deposit = q / length

        for cell in route:
            pheromones[cell] += deposit

    pheromones[:] = np.clip(pheromones, 0.001, 100)

    st.session_state.pheromones = pheromones

    st.session_state.history.append({
        "iteration": st.session_state.iteration,
        "meilleure_distance": (
            st.session_state.best_length
            if math.isfinite(st.session_state.best_length)
            else None
        ),
        "fourmis_arrivees": len(
            st.session_state.completed_paths
        )
    })


def next_iteration(rho, q):
    update_pheromones(rho, q)

    st.session_state.iteration += 1

    st.session_state.ants = []

    for i in range(st.session_state.nb_ants):
        st.session_state.ants.append({
            "id": i + 1,
            "position": NEST,
            "path": [NEST],
            "phase": "aller",
            "status": "active",
            "length": 0.0,
            "steps": 0,
            "color": plt.cm.tab20(i % 20),
        })

    st.session_state.completed_paths = []
    st.session_state.step = 0
    st.session_state.stage = "construction"
    st.session_state.message = (
        "Nouvelle itération : les fourmis repartent du nid."
    )


# ============================================================
# DRAWING
# ============================================================

def draw_ant(ax, position, color, phase="aller"):
    r, c = position

    # Ant body: abdomen, thorax and head.
    ax.add_patch(Ellipse(
        (c - 0.18, -r),
        0.34, 0.24,
        color=color,
        zorder=8
    ))

    ax.add_patch(Circle(
        (c + 0.03, -r),
        0.10,
        color=color,
        zorder=9
    ))

    ax.add_patch(Circle(
        (c + 0.19, -r),
        0.12,
        color=color,
        zorder=10
    ))

    # Antennae.
    ax.plot(
        [c + 0.23, c + 0.30],
        [-r + 0.08, -r + 0.20],
        color="black",
        linewidth=1.4,
        zorder=11
    )

    ax.plot(
        [c + 0.23, c + 0.37],
        [-r + 0.07, -r + 0.15],
        color="black",
        linewidth=1.4,
        zorder=11
    )

    # Six legs.
    for offset in [-0.09, 0.0, 0.09]:
        ax.plot(
            [c + offset, c + offset - 0.12],
            [-r, -r - 0.15],
            color="black",
            linewidth=1.2,
            zorder=7
        )

        ax.plot(
            [c + offset, c + offset - 0.12],
            [-r, -r + 0.15],
            color="black",
            linewidth=1.2,
            zorder=7
        )


def draw_environment():
    fig, ax = plt.subplots(figsize=(13, 8))

    pheromones = st.session_state.pheromones.copy()

    # Pheromone background.
    display = pheromones.copy()

    for cell in OBSTACLES:
        display[cell] = np.nan

    ax.imshow(
        display,
        cmap="YlOrBr",
        alpha=0.55,
        interpolation="nearest",
        origin="upper"
    )

    # Grid.
    ax.set_xticks(np.arange(-0.5, COLS, 1))
    ax.set_yticks(np.arange(-0.5, ROWS, 1))
    ax.grid(color="gray", alpha=0.25)
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)

    # Obstacles.
    for r, c in OBSTACLES:
        ax.add_patch(plt.Rectangle(
            (c - 0.5, r - 0.5),
            1, 1,
            color="#555555",
            zorder=3
        ))

    # Nest.
    ax.add_patch(Circle(
        (NEST[1], NEST[0]),
        0.42,
        color="#8B4513",
        zorder=5
    ))
    ax.text(
        NEST[1], NEST[0] - 0.7,
        "NID",
        ha="center",
        weight="bold",
        color="#8B4513"
    )

    # Food.
    ax.add_patch(Circle(
        (FOOD[1], FOOD[0]),
        0.42,
        color="gold",
        edgecolor="darkorange",
        linewidth=2,
        zorder=5
    ))
    ax.text(
        FOOD[1], FOOD[0] - 0.7,
        "NOURRITURE",
        ha="center",
        weight="bold",
        color="#B8860B"
    )

    # Best path.
    if st.session_state.best_path:
        path = st.session_state.best_path
        xs = [cell[1] for cell in path]
        ys = [-cell[0] for cell in path]

        ax.plot(
            xs, ys,
            color="limegreen",
            linewidth=2.8,
            alpha=0.9,
            zorder=6,
            label="Meilleur chemin"
        )

    # Current ants and their paths.
    for ant in st.session_state.ants:
        if ant["status"] == "active":
            path = ant["path"]

            if len(path) > 1:
                xs = [cell[1] for cell in path]
                ys = [-cell[0] for cell in path]

                ax.plot(
                    xs, ys,
                    color=ant["color"],
                    linewidth=1.4,
                    alpha=0.65,
                    zorder=6
                )

            draw_ant(
                ax,
                ant["position"],
                ant["color"],
                ant["phase"]
            )

    ax.set_title(
        "Recherche de nourriture par les fourmis artificielles",
        fontsize=15,
        weight="bold"
    )

    ax.set_xlabel("Colonnes")
    ax.set_ylabel("Lignes")

    ax.invert_yaxis()
    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Paramètres")

nb_ants = st.sidebar.slider(
    "Nombre de fourmis",
    1, 15, 5
)

alpha = st.sidebar.slider(
    "Alpha (importance des phéromones)",
    0.0, 5.0, 1.0, 0.1
)

beta = st.sidebar.slider(
    "Beta (importance de l'information heuristique)",
    0.0, 5.0, 2.0, 0.1
)

rho = st.sidebar.slider(
    "Taux d'évaporation ρ",
    0.0, 0.9, 0.2, 0.05
)

q = st.sidebar.slider(
    "Quantité de phéromone Q",
    1.0, 100.0, 20.0
)

if st.sidebar.button("🔄 Réinitialiser la simulation"):
    st.session_state.nb_ants = nb_ants
    initialize()
    st.session_state.initialized = True
    st.rerun()

if not st.session_state.initialized:
    st.session_state.nb_ants = nb_ants
    initialize()
    st.session_state.initialized = True


# ============================================================
# MAIN INTERFACE
# ============================================================

st.title("🐜 Algorithme des colonies de fourmis")
st.subheader("Simulation : recherche de nourriture")

st.markdown("""
Cette simulation illustre comment plusieurs fourmis artificielles
peuvent trouver un chemin entre leur nid et une source de nourriture.

Chaque fourmi prend ses décisions en fonction de :
- **La phéromone (τ)** : trace laissée par les fourmis précédentes.
- **L'information heuristique (η)** : proximité de la destination.
- **La probabilité (P)** : chance de choisir une direction.
""")

# ============================================================
# BUTTONS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    next_step = st.button(
        "▶ Étape suivante",
        use_container_width=True
    )

with col2:
    run_iteration = st.button(
        "⏩ Exécuter une itération",
        use_container_width=True
    )

with col3:
    run_all = st.button(
        "🚀 Lancer 5 itérations",
        use_container_width=True
    )

if next_step:
    active_ants = [
        ant for ant in st.session_state.ants
        if ant["status"] == "active"
    ]

    if active_ants:
        ant = active_ants[0]
        move_ant(ant, alpha, beta)
        st.session_state.step += 1
        st.session_state.stage = "construction"
    else:
        next_iteration(rho, q)

if run_iteration or run_all:
    iterations_to_run = 1 if run_iteration else 5

    for _ in range(iterations_to_run):
        for _step in range(ROWS * COLS * 2):
            active_ants = [
                ant for ant in st.session_state.ants
                if ant["status"] == "active"
            ]

            if not active_ants:
                break

            for ant in active_ants:
                move_ant(ant, alpha, beta)

        next_iteration(rho, q)

st.info(st.session_state.message)

# ============================================================
# ENVIRONMENT AND STATISTICS
# ============================================================

left, right = st.columns([2, 1])

with left:
    st.pyplot(draw_environment(), use_container_width=True)

with right:
    st.markdown("### 📊 État de la simulation")

    st.metric(
        "Itération",
        st.session_state.iteration
    )

    st.metric(
        "Meilleure distance",
        (
            f"{st.session_state.best_length:.2f}"
            if math.isfinite(st.session_state.best_length)
            else "Pas encore trouvée"
        )
    )

    st.metric(
        "Fourmis revenues au nid",
        sum(
            ant["status"] == "terminée"
            for ant in st.session_state.ants
        )
    )

    st.markdown("### 🐜 État des fourmis")

    ant_table = []

    for ant in st.session_state.ants:
        ant_table.append({
            "Fourmi": ant["id"],
            "Position": str(ant["position"]),
            "Phase": ant["phase"],
            "État": ant["status"],
            "Distance": round(ant["length"], 2)
        })

    st.dataframe(
        pd.DataFrame(ant_table),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# CALCULATION DETAILS
# ============================================================

st.divider()
st.header("🧮 Calcul de la décision")

if st.session_state.last_candidates:
    st.markdown(
        f"**Fourmi active :** {st.session_state.current_ant}"
    )

    calculation_rows = []

    for item in st.session_state.last_candidates:
        calculation_rows.append({
            "Position candidate": str(item["cell"]),
            "Distance d": round(item["distance"], 3),
            "Phéromone τ": round(item["tau"], 4),
            "Heuristique η": round(item["eta"], 4),
            "Poids τᵅ × ηᵝ": round(item["weight"], 5),
            "Probabilité": round(item["probability"], 4)
        })

    st.dataframe(
        pd.DataFrame(calculation_rows),
        use_container_width=True,
        hide_index=True
    )

    if st.session_state.last_decision:
        selected = st.session_state.last_decision

        st.success(
            f"Direction choisie : {selected['cell']} "
            f"| Probabilité : {selected['probability']:.2%}"
        )
else:
    st.info(
        "Cliquez sur « Étape suivante » pour observer "
        "la première décision d'une fourmi."
    )

# ============================================================
# PHEROMONE MATRIX
# ============================================================

st.divider()
st.header("🌡️ Matrice des phéromones")

st.caption(
    "Chaque cellule représente la quantité de phéromone "
    "associée à une position de l'environnement. "
    "Il s'agit d'une simplification pédagogique du modèle."
)

pheromone_df = pd.DataFrame(
    st.session_state.pheromones
)

st.dataframe(
    pheromone_df.round(2),
    use_container_width=True
)

# ============================================================
# PSEUDOCODE
# ============================================================

st.divider()
st.header("📜 Pseudocode de l'algorithme ACO")

pseudocode = [
    ("1", "Initialiser les phéromones", "Initialisation"),
    ("2", "Placer les fourmis au nid", "Construction"),
    ("3", "Identifier les déplacements possibles", "Construction"),
    ("4", "Calculer τ et η pour chaque direction", "Calcul"),
    ("5", "Calculer les probabilités de transition", "Calcul"),
    ("6", "Choisir une direction aléatoirement", "Décision"),
    ("7", "Déplacer la fourmi", "Construction"),
    ("8", "Répéter jusqu'à atteindre la nourriture", "Construction"),
    ("9", "Retourner au nid", "Construction"),
    ("10", "Évaluer la longueur du chemin", "Évaluation"),
    ("11", "Évaporer les phéromones", "Amélioration"),
    ("12", "Renforcer les meilleurs chemins", "Amélioration"),
    ("13", "Répéter les itérations", "Terminaison")
]

for number, description, stage in pseudocode:
    if stage == st.session_state.stage:
        st.markdown(
            f"🟢 **{number}. {description}**"
            "  ← Étape actuelle"
        )
    else:
        st.markdown(
            f"⚪ {number}. {description}"
        )

# ============================================================
# FORMULAS
# ============================================================

st.divider()
st.header("📐 Formules utilisées")

st.latex(
    r"P_{ij} = "
    r"\frac{\tau_{ij}^{\alpha}\eta_{ij}^{\beta}}"
    r"{\sum_{l \in N_i}\tau_{il}^{\alpha}\eta_{il}^{\beta}}"
)

st.latex(
    r"\eta_{ij} = \frac{1}{d(j,\text{destination})}"
)

st.latex(
    r"\tau_{ij} \leftarrow (1-\rho)\tau_{ij}"
)

st.latex(
    r"\tau_{ij} \leftarrow \tau_{ij} + "
    r"\frac{Q}{L_k}"
)

st.markdown("""
**Interprétation :**

- α contrôle l'influence des phéromones.
- β contrôle l'influence de l'information heuristique.
- ρ représente le taux d'évaporation.
- Q représente la quantité de phéromone déposée.
- Lₖ représente la longueur du chemin parcouru par la fourmi k.

L'évaporation réduit progressivement les traces anciennes.
Le renforcement favorise les chemins courts découverts par les fourmis.
""")
