"""Taxonomía de clases: posiciones -> clases finales -> bloques.

Debe mantenerse en sincronía con las definiciones de los notebooks 04 y siguientes.
"""

MAPA_CLASE = {
    "CB": "Central",
    "LB": "Lateral", "RB": "Lateral",
    "LWB": "Carrilero", "RWB": "Carrilero",
    "CDM": "Pivote", "CM": "Interior", "CAM": "Mediapunta",
    "LM": "Mediocampista de banda", "RM": "Mediocampista de banda",
    "LW": "Extremo", "RW": "Extremo",
    "ST": "Delantero centro", "CF": "Delantero centro",
}

ORDEN = ["Central", "Lateral", "Carrilero", "Pivote", "Interior", "Mediapunta",
         "Mediocampista de banda", "Extremo", "Delantero centro"]

BLOQUES = {
    "A": {"Central": "Defensa", "Lateral": "Defensa", "Carrilero": "Defensa",
          "Pivote": "Medio", "Interior": "Medio", "Mediapunta": "Medio",
          "Mediocampista de banda": "Medio",
          "Extremo": "Delantero", "Delantero centro": "Delantero"},
    "B": {"Central": "Defensa", "Lateral": "Defensa", "Carrilero": "Defensa",
          "Pivote": "Medio", "Interior": "Medio", "Mediapunta": "Medio",
          "Mediocampista de banda": "Banda", "Extremo": "Banda",
          "Delantero centro": "Delantero"},
}
