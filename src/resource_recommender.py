import pandas as pd
import logging

logger = logging.getLogger(__name__)

def recommend_resources(risk_level: str, event_cause: str, requires_road_closure: bool) -> dict:
    """
    Recommend traffic management resources based on risk level, event cause, and road closure needs.
    
    Parameters:
    risk_level (str): 'Low', 'Medium', 'High', 'Critical'.
    event_cause (str): Cause of the incident (e.g. accident, breakdown, tree_fall).
    requires_road_closure (bool): Whether a road closure is required.
    
    Returns:
    dict: Recommended counts and emergency requirements.
    """
    # 1. Traffic Personnel Allocation (Min, Max)
    personnel_map = {
        'Low': (1, 2),
        'Medium': (3, 5),
        'High': (6, 9),
        'Critical': (10, 15)
    }
    
    min_p, max_p = personnel_map.get(risk_level, (2, 4))
    
    # Road closures require additional staffing for diversion points
    if requires_road_closure:
        min_p += 2
        max_p += 3
        
    # 2. Barricade Count Allocation (Min, Max)
    if requires_road_closure:
        # Road closures require a significant barrier setup
        barricade_map = {
            'Low': (10, 15),
            'Medium': (15, 20),
            'High': (20, 25),
            'Critical': (25, 40)
        }
        min_b, max_b = barricade_map.get(risk_level, (15, 25))
    else:
        barricade_map = {
            'Low': (0, 2),
            'Medium': (3, 6),
            'High': (8, 12),
            'Critical': (15, 25)
        }
        min_b, max_b = barricade_map.get(risk_level, (2, 5))
        
    # 3. Traffic Control Units (Patrol vehicles / Interceptors)
    unit_map = {
        'Low': 0,
        'Medium': 1,
        'High': 2,
        'Critical': 3
    }
    units = unit_map.get(risk_level, 1)
    
    # 4. Emergency Response Requirements
    emergency_reqs = []
    cause_clean = str(event_cause).lower().strip()
    
    if cause_clean == 'accident':
        emergency_reqs.append("Ambulance (High Priority - Immediate dispatch for medical aid)")
        emergency_reqs.append("Heavy Tow Truck (For clearing accident vehicles from lanes)")
        emergency_reqs.append("Police Patrol (For scene documentation & investigation)")
    elif cause_clean == 'vehicle_breakdown':
        emergency_reqs.append("Tow Truck / Recovery Vehicle (Standard dispatch to tow disabled vehicle)")
        emergency_reqs.append("Traffic Patrol Vehicle (For local lane channeling and warnings)")
    elif cause_clean == 'tree_fall':
        emergency_reqs.append("Forestry/Tree Disposal Squad (Equipped with chainsaws and woodchippers)")
        emergency_reqs.append("Emergency Clean-up Team (For immediate branch and leaf debris removal)")
        emergency_reqs.append("Heavy Tow Truck (Standby for clearing damaged vehicles)")
    elif cause_clean == 'water_logging':
        emergency_reqs.append("Municipal Pumping & Drainage Squad (Equipped with high-capacity pumps)")
        emergency_reqs.append("Barricade Deployment Team (To close submerged lanes or underpasses)")
    elif cause_clean == 'construction':
        emergency_reqs.append("Signage & Coning Team (For safety channeling and lane merges)")
        emergency_reqs.append("Routine Traffic Patrol (For daily construction site safety checks)")
    elif cause_clean == 'public_event':
        emergency_reqs.append("Event Security Liaison Squad (For coordinate with event organizers)")
        emergency_reqs.append("Crowd Control Barriers Team")
        emergency_reqs.append("Emergency Medical Standby (Ambulance on-site)")
    elif cause_clean == 'congestion':
        emergency_reqs.append("Q-Management Interceptor Unit")
        emergency_reqs.append("Traffic Control Room Liaison (For signal timing adjustments)")
    else:
        # Default or others
        if risk_level in ['High', 'Critical']:
            emergency_reqs.append("General Emergency Response Patrol")
            
    if not emergency_reqs:
        emergency_reqs.append("None (Standard traffic patrol sufficient)")
        
    return {
        'traffic_personnel_min': min_p,
        'traffic_personnel_max': max_p,
        'barricades_min': min_b,
        'barricades_max': max_b,
        'traffic_control_units': units,
        'emergency_response': emergency_reqs
    }

def recommend_resources_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply resource recommendations across an entire DataFrame.
    Fix 5.2: uses df.apply() instead of iterrows() for better performance.
    """
    logger.info("Generating resource recommendations for dataset...")
    df = df.copy()

    def _apply_row(row):
        rec = recommend_resources(
            risk_level=row['risk_level'],
            event_cause=row['event_cause'],
            requires_road_closure=row['requires_road_closure'],
        )
        # Flatten emergency_response list to string for CSV compatibility
        rec['emergency_response'] = "; ".join(rec['emergency_response'])
        return pd.Series(rec)

    rec_df = df.apply(_apply_row, axis=1)

    for col in rec_df.columns:
        df[col] = rec_df[col]

    return df

