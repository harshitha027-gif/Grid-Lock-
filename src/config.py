"""
Centralized configuration for the Astram Traffic Intelligence Platform.
Single source of truth for all scoring weights, maps, and thresholds.
Fixes Issue 2.1: scoring formula was duplicated in 3 places.
"""

# ─── Risk Score Weights (must sum to 1.0) ───────────────────────────────────
RISK_WEIGHTS = {
    'road_closure': 0.25,
    'priority':     0.20,
    'cause':        0.20,
    'duration':     0.20,
    'density':      0.15,
}

# ─── Priority → Score mapping ────────────────────────────────────────────────
PRIORITY_SCORE_MAP = {
    'High':   100.0,
    'Medium':  60.0,
    'Low':     25.0,
}

# ─── Event Cause → Base severity score ───────────────────────────────────────
CAUSE_SCORE_MAP = {
    'accident':          100.0,
    'water_logging':     100.0,
    'congestion':        100.0,
    'public_event':       85.0,
    'construction':       85.0,
    'tree_fall':          85.0,
    'vehicle_breakdown':  65.0,
    'pot_holes':          45.0,
    'road_conditions':    45.0,
}
CAUSE_SCORE_DEFAULT = 25.0        # For any cause not in the map
UNPLANNED_BONUS = 5.0             # Added to cause score for unplanned events

# ─── Duration scaling ────────────────────────────────────────────────────────
DURATION_MAX_MINUTES = 240.0      # 4 hours → duration_score = 100

# ─── Risk Level thresholds (score < threshold → level) ───────────────────────
# Evaluated in order; first match wins.
RISK_THRESHOLDS = [
    (35.0,  'Low'),
    (55.0,  'Medium'),
    (75.0,  'High'),
    (float('inf'), 'Critical'),
]

# ─── Bengaluru coordinate bounding box (for validation) ──────────────────────
BENGALURU_LAT_RANGE = (12.5, 13.5)
BENGALURU_LON_RANGE = (77.2, 77.9)

# ─── Spatial density radius (km) ─────────────────────────────────────────────
DENSITY_RADIUS_KM = 0.5
EARTH_RADIUS_KM = 6371.0
