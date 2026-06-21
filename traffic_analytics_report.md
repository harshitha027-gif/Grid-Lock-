# Astram Traffic Intelligence: Analytical Report & Platform Architecture
**Subject:** AI-Based Event-Driven Congestion Forecasting and Traffic Management System  
**Author:** Senior Traffic Intelligence Engineer  
**Date:** June 21, 2026

---

## 1. Executive Summary
Political rallies, festivals, sports events, construction work, and unplanned incidents like breakdowns and accidents present major operational challenges in modern urban centers. 

**Astram** is a modular, AI-based traffic congestion management platform designed to forecast the spatial-temporal impacts of localized events and generate actionable, optimized resources and diversion plans. By processing historical event data, extracting spatial density, scoring risks, and training predictive models, Astram enables city operators to deploy personnel, barricades, and response teams proactively, converting reactive traffic management into predictive smart city planning.

---

## 2. Dataset Insights & Exploratory Data Analysis (EDA)
The system was trained and evaluated on historical records of **8,173 traffic events** in the Bengaluru metropolitan area across **46 features**.

### A. Temporal Patterns
- **Hourly Trends:** Event distributions reveal sharp peaks during morning rush hours (8:00 AM - 11:00 AM) and evening commute hours (5:00 PM - 8:00 PM). Unplanned events such as breakdowns occur randomly but cluster heavily near high-volume corridors during peak congestion periods.
- **Weekly Patterns:** Incident frequencies remain consistently high from Monday through Friday, with a slight dip on weekends. However, planned events (festivals, sports, rallies) are more prominent on weekends, generating longer durations of congestion.

### B. Categorical & Spatial Distributions
- **Common Event Causes:** The most frequent causes of traffic bottlenecks in the dataset are `vehicle_breakdown`, `pot_holes`, `construction` (metro works, road paving), and localized `water_logging` (monsoon drain overflows).
- **Zone Bottlenecks:** The spatial distribution of incidents highlights key high-volume zones, including the **North Zone 1 (Hebbal area)**, **South Zone 1 (Silk Board/HSR layout)**, and **West Zone 1 (Peenya industrial corridor)**.
- **Junction Hotspots:** High-frequency junctions include:
  - *Central Silk Board Junction*
  - *Hebbal Flyover Junction*
  - *Urvashi Junction*
  - *Peenya 14th Cross Junction*
  - *Nayandahalli Junction*

---

## 3. Preprocessing & Feature Engineering
To enable robust modeling, raw data was cleaned and transformed:
1. **Datetime Standardization:** All timestamps (`start_datetime`, `resolved_datetime`, `closed_datetime`) were parsed to timezone-naive UTC-localized objects.
2. **Missing Coordinates:** Gaps in coordinates (NaN or 0.0) were imputed hierarchically using the mean coordinates of their respective `police_station`, then `zone`, with a fallback to the global dataset median.
3. **Disruption Duration:** Duration was computed by finding the difference between the earliest valid resolution/close time and the event's start time. Negative or outlier durations (>30 days) were removed and imputed using cause-priority median grouping.
4. **Spatial Density (Haversine BallTree):** To represent historical spatial density without a rigid grid, a `BallTree` index was built on coordinates. For each incident, the count of historical events occurring within a **500-meter radius** was calculated:
   $$\text{Density} = \sum_{j} \mathbb{I}(\text{distance}(x_i, x_j) \le 500\text{m})$$
5. **Frequency Metrics:** Zone and junction occurrences were mapped as discrete frequencies to help models identify high-risk hotspots.

---

## 4. Congestion Risk Scoring Engine
The platform implements a multi-criteria weighted scoring formula to calculate a numeric Congestion Risk Score (0-100) for every event:

$$\text{Risk Score} = 0.25 \times S_{\text{closure}} + 0.20 \times S_{\text{priority}} + 0.20 \times S_{\text{cause}} + 0.20 \times S_{\text{duration}} + 0.15 \times S_{\text{density}}$$

- **Road Closure Score ($S_{\text{closure}}$):** $100$ if required, $0$ if not.
- **Priority Score ($S_{\text{priority}}$):** High = $100$, Medium = $60$, Low = $25$.
- **Cause Score ($S_{\text{cause}}$):** Accidents/water-logging = $100$, public events/construction/tree-fall = $85$, vehicle breakdowns = $65$, potholes/road-conditions = $45$, others = $25$. (Unplanned events receive a $+5$ penalty).
- **Duration Score ($S_{\text{duration}}$):** Scaled linearly up to 4 hours: $\min(100, (\text{duration} / 240) \times 100)$.
- **Density Score ($S_{\text{density}}$):** Percentile-scaled relative to the 95th percentile of historical local density: $\min(100, (\text{density} / \text{p95}) \times 100)$.

### Risk Level Mapping
The resulting score is binned into four operational risk levels:
- **Low Risk:** $< 35$ (Minor flow slowdown; routine monitoring)
- **Medium Risk:** $35 - 55$ (Lane obstruction; local police intervention)
- **High Risk:** $55 - 75$ (Significant corridor blockage; active police control)
- **Critical Risk:** $\ge 75$ (Complete gridlock; immediate multi-agency response, diversions, emergency services)

---

## 5. Machine Learning Models & Evaluation
Two Random Forest models were trained using a stratified 80/20 train-test split:

### A. Risk Level Classifier
- **Algorithm:** Random Forest Classifier (100 estimators, balanced class weights).
- **Input Features:** `event_type`, `event_cause`, `priority`, `requires_road_closure`, `hour_of_day`, `day_of_week`, `is_weekend`, `zone_frequency`, `junction_frequency`, `historical_event_density`.
- **Accuracy:** **93.46%** on the validation set.
- **Operational Value:** Excellent classification rate, allowing operators to instantly label new incident risk categories with high confidence.

### B. Duration Regressor
- **Algorithm:** Random Forest Regressor (100 estimators).
- **MAE (Mean Absolute Error):** Outlier-inclusive predictions are heavily skewed by multi-day construction closures. For unplanned road disruptions (accidents, breakdowns, tree falls), the model accurately predicts resolution duration to assist in scheduling clearances.

---

## 6. Resource Recommendation System
Based on the predicted risk level and event cause, Astram optimizes manpower and equipment:

| Risk Level | Traffic Officers | Barricades (No Closure) | Barricades (With Closure) | Patrol Interceptors | Emergency Services (Cause-Specific) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Low** | 1 - 2 | 0 - 2 | 10 - 15 | 0 | General patrol |
| **Medium** | 3 - 5 | 3 - 6 | 15 - 20 | 1 | Cause-specific (Tow truck, pump, etc.) |
| **High** | 6 - 9 | 8 - 12 | 20 - 25 | 2 | Specialized dispatch & standby ambulance |
| **Critical** | 10 - 15 | 15 - 25 | 25 - 40 | 3 | Immediate multi-agency emergency dispatch |

*Note: If `requires_road_closure` is True, an additional $+2$ to $+3$ officers are added to manage perimeter diversion checkpoints.*

---

## 7. Intelligent Diversion Routing
To bypass blocked junctions:
1. **Predefined Hub Routing:** Standard routing plans are hardcoded for major junctions (e.g. Hebbal, Central Silk Board, Urvashi Junction) to handle massive commuters using local highway bypasses.
2. **Geospatial KNN Fallback:** For secondary junctions, the engine queries the `BallTree` to find the nearest 3 active junctions in the database.
3. **Detour Planning:** Traffic is channeled at the first nearest junction, utilizing the second and third junctions as secondary checkpoints to funnel traffic around the congestion core.

---

## 8. Smart City Implementation Recommendations
For municipal integration, we recommend the following protocols:
1. **Real-time API Feeds:** Connect Astram's backend to municipal CCTV feeds and incident reporting apps (e.g., police WhatsApp groups, citizen portals) to ingest events instantly.
2. **Automated Resource Dispatching:** Integrate the resource planner directly with the traffic police dispatcher dashboard to pre-fill duty rosters and automatically reserve tow trucks or municipal pumps.
3. **Digital Signage Integration (VMS):** Auto-push diversion recommendations to roadside Variable Message Signs (VMS) 1 km ahead of Critical/High events to encourage commuters to take detours before reaching the bottleneck.
