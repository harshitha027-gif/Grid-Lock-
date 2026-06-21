import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import logging

logger = logging.getLogger(__name__)

# Predefined diversion strategies for famous traffic hotspots/junctions in Bengaluru
PREDEFINED_DIVERSIONS = {
    'HebbalFlyoverJunc': (
        "Hebbal Flyover Junction Blockage Strategy:\n"
        "1. Outbound Traffic (towards Airport): Divert at Sanjay Nagar Main Road and route via Bellary Road service lanes.\n"
        "2. Inbound Traffic (towards City): Divert traffic at Outer Ring Road (ORR) junction towards Hennur/Banaswadi.\n"
        "3. Alternate Route: Use Thanisandra Main Road as an alternative corridor to reach ORR."
    ),
    'SilkBoardJunc': (
        "Central Silk Board Junction Blockage Strategy:\n"
        "1. BTM to HSR Traffic: Divert at Madiwala Underpass towards Hosur Road / 27th Main.\n"
        "2. Electronic City to City Traffic: Divert at Bommanahalli Junction towards Begur Road / Bannerghatta Road.\n"
        "3. Alternate Route: Use the Nice Ring Road or HSR Sector 1 sector roads to bypass the main junction."
    ),
    'UrvashiJunction': (
        "Urvashi Junction Blockage Strategy:\n"
        "1. Lalbagh Road Traffic: Divert via JC Road and KH Road.\n"
        "2. Richmond Town Traffic: Divert via Hosur Road and Langford Road.\n"
        "3. Alternate Route: Use Siddaiah Road for light vehicle bypass."
    ),
    'Peenya14thCrossJunc': (
        "Peenya Industrial Area 14th Cross Blockage Strategy:\n"
        "1. Heavy Vehicles: Divert at Goraguntepalya Junction towards Outer Ring Road / Tumkur Road bypass.\n"
        "2. Light Vehicles: Redirect through Peenya 1st Stage and NTTF Road interior corridors.\n"
        "3. Alternate Route: Use Jalahalli Cross for east-west movement."
    ),
    'MysoreRd-RingRdJunc(Nayandanahallii)': (
        "Nayandahalli Junction Blockage Strategy:\n"
        "1. ORR Traffic (towards JP Nagar): Divert at Gnanabharathi towards Ullal Road.\n"
        "2. Mysore Road Inbound: Divert at Pantarapalya towards Chord Road / Vijayanagar.\n"
        "3. Alternate Route: Route traffic through Kengeri Satellite Town roads."
    ),
    'LalbaghMainGateJunc': (
        "Lalbagh Main Gate Junction Blockage Strategy:\n"
        "1. Central Business District (CBD) bound traffic: Divert via Double Road (KH Road) towards Richmond Circle.\n"
        "2. South-bound traffic: Route via Lalbagh Fort Road and Kanakapura Road.\n"
        "3. Alternate Route: Divert at Urvashi Junction towards JC Road."
    ),
    'QueensStatueCircle': (
        "Queens Statue Circle Blockage Strategy:\n"
        "1. MG Road Traffic: Divert at Anil Kumble Circle towards Cubbon Road / Infantry Road.\n"
        "2. Kasturba Road Traffic: Route via Hudson Circle and Devanga Hostel Road.\n"
        "3. Alternate Route: Use Lavelle Road for one-way light vehicle passage."
    )
}

class DiversionEngine:
    def __init__(self, historical_df: pd.DataFrame = None):
        """
        Initialize the diversion engine with historical data to learn unique junctions and locations.
        """
        self.junction_coords = pd.DataFrame()
        self.ball_tree = None
        
        if historical_df is not None and not historical_df.empty:
            self.fit(historical_df)
            
    def fit(self, df: pd.DataFrame):
        """
        Extract unique junctions and build the BallTree.
        """
        logger.info("Fitting Diversion Engine on historical junctions...")
        
        # Clean and filter junctions
        junction_df = df[(df['junction'].notna()) & (df['junction'] != 'Unknown') & (df['junction'] != 'NULL')].copy()
        
        if junction_df.empty:
            logger.warning("No valid junctions found in the historical data to fit.")
            return
            
        # Group by junction and calculate average latitude/longitude
        self.junction_coords = junction_df.groupby('junction')[['latitude', 'longitude']].mean().reset_index()
        
        # Build spatial BallTree for nearest neighbor lookups
        coords_rad = np.radians(self.junction_coords[['latitude', 'longitude']].values)
        self.ball_tree = BallTree(coords_rad, metric='haversine')
        logger.info(f"Diversion Engine fitted with {len(self.junction_coords)} unique junctions.")
        
    def recommend_diversion(self, latitude: float, longitude: float, zone: str = 'Unknown', junction_name: str = 'Unknown') -> dict:
        """
        Suggest diversion route instructions and nearest junctions for diversion points.
        """
        # 1. Check if there is a predefined strategy for this exact junction
        j_clean = str(junction_name).strip()
        if j_clean in PREDEFINED_DIVERSIONS:
            strategy = PREDEFINED_DIVERSIONS[j_clean]
            is_predefined = True
        else:
            strategy = ""
            is_predefined = False
            
        # 2. Find nearest junctions using BallTree
        nearest_junctions = []
        
        if self.ball_tree is not None and not self.junction_coords.empty:
            # Query point
            query_pt = np.radians([[float(latitude), float(longitude)]])
            
            # Find nearest 4 in case the nearest is the event junction itself
            k_val = min(4, len(self.junction_coords))
            distances, indices = self.ball_tree.query(query_pt, k=k_val)
            
            # Earth radius in km
            earth_radius_km = 6371.0
            
            for dist, idx in zip(distances[0], indices[0]):
                j_info = self.junction_coords.iloc[idx]
                dist_km = dist * earth_radius_km
                
                # Exclude if it's the exact same junction name (unless it's unknown)
                if j_clean != 'Unknown' and j_info['junction'] == j_clean:
                    continue
                    
                nearest_junctions.append({
                    'junction': j_info['junction'],
                    'latitude': j_info['latitude'],
                    'longitude': j_info['longitude'],
                    'distance_km': round(dist_km, 2)
                })
                
        # Limit to top 3 nearest alternative junctions
        nearest_junctions = nearest_junctions[:3]
        
        # 3. Create a dynamic text strategy if no predefined strategy exists
        if not is_predefined:
            if len(nearest_junctions) >= 2:
                j1 = nearest_junctions[0]['junction']
                d1 = nearest_junctions[0]['distance_km']
                j2 = nearest_junctions[1]['junction']
                d2 = nearest_junctions[1]['distance_km']
                
                j3_text = ""
                if len(nearest_junctions) == 3:
                    j3 = nearest_junctions[2]['junction']
                    d3 = nearest_junctions[2]['distance_km']
                    j3_text = f" and redirect overflow towards **{j3}** ({d3} km away)"
                
                strategy = (
                    f"Dynamic Diversion Plan ({zone} Zone):\n"
                    f"1. Divert traffic at the nearest intersection: **{j1}** (located {d1} km away).\n"
                    f"2. Set up a primary detour checkpoint at **{j2}** (located {d2} km away){j3_text}.\n"
                    f"3. Signage Notice: Place digital boards 500m ahead of the incident to advise drivers to take alternate routes through these diversion points."
                )
            else:
                strategy = (
                    f"General Diversion Plan ({zone} Zone):\n"
                    f"1. Alert local traffic control to place diversion markers 200 meters prior to the event coordinate.\n"
                    f"2. Request manual traffic control at nearby major intersections to manage queue dispersion."
                )
                
        return {
            'is_predefined': is_predefined,
            'diversion_plan': strategy,
            'nearest_diversion_junctions': nearest_junctions
        }
