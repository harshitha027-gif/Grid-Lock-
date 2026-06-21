# Project Issues Report — Astram Event-Driven Congestion Platform

## 1. Critical Model Quality Issues

### 1.1 Duration Regressor Is Essentially Useless
- **File:** `models/model_metrics.json`
- **Problem:** Regressor R² = **0.209** (explains only ~21% of variance). MAE = **1549 minutes (~25.8 hours)**. A model predicting median duration every time would likely outperform this.
- **Impact:** Tab 2 "Forecasted Disruption Duration" outputs are unreliable. Users cannot trust duration predictions for operational planning.
- **Root Cause:** Duration in the dataset is heavily imputed (preprocessing fills NaN from group medians). ML model learns imputed medians back — circular signal. Real variance is lost.
- **Fix:** Collect actual end timestamps. Remove imputed records from regressor training (use `valid_duration_mask` strictly). Consider predicting duration buckets (< 30min, 30-120min, > 120min) as classification instead.

### 1.2 Risk Classifier Is Trained on Its Own Labels
- **Files:** `src/risk_engine.py:96-113`, `run_pipeline.py:46-55`
- **Problem:** `train_models()` calls `calculate_risk_score()` internally to **generate labels**, then trains classifier to predict those same labels. The classifier learns to replicate a deterministic formula — not real-world risk.
- **Impact:** 93.4% accuracy is **meaningless** — the classifier just approximates a formula that is already applied directly at inference in `predict_event()`. Two competing systems (formula + ML) return different results for same input.
- **Fix:** Either: (a) drop the ML classifier entirely and use the scoring formula only, OR (b) define ground truth labels from expert annotation or outcome data (e.g., actual road closures, police dispatch logs), then train on those.

### 1.3 Data Leakage in Feature Engineering (Training Mode)
- **File:** `src/feature_engineering.py:30-59`
- **Problem:** `zone_frequency`, `junction_frequency`, and `historical_event_density` are computed from the **entire df** (including test split rows) when `is_training=True`. BallTree is built on all rows before train/test split.
- **Impact:** Test set performance is inflated. Real inference on unseen events will see different frequency/density values.
- **Fix:** Compute frequency maps on training split only, then map to test split. Do the split before calling `engineer_features`.

---

## 2. Logical / Design Issues

### 2.1 Duplicate Risk Score Computation (Code Duplication)
- **Files:** `src/risk_engine.py:257-284`, `app.py:532-565`
- **Problem:** The weighted scoring formula (road closure 25%, priority 20%, cause 20%, duration 20%, density 15%) is copy-pasted in three places: `calculate_risk_score()`, `predict_event()`, and the fallback block in `app.py`. Any weight change must be updated in all three.
- **Fix:** Extract scoring weights into a single `RISK_WEIGHTS` constant dict. Call one shared function everywhere.

### 2.2 `predict_event()` Uses ML Model for Level but Formula for Score
- **File:** `src/risk_engine.py:253-296`
- **Problem:** `predicted_risk_level` comes from ML classifier, but `risk_score` is recomputed via the hand-crafted formula using `predicted_duration`. These two can disagree — e.g., ML says "High" but formula score = 38 (Medium threshold).
- **Impact:** UI shows inconsistent level vs. score gauge. Users cannot trust either.
- **Fix:** Align: either derive level from formula score (no ML), or have ML output probabilities and use those for score.

### 2.3 `zone_frequency` / `junction_frequency` Hardcoded at Inference
- **File:** `src/risk_engine.py:239-242`
- **Problem:** At inference, if these columns are missing, they are set to `100` and `10` (arbitrary "reasonable averages"). These values have no relation to the actual event location's historical frequency.
- **Impact:** Density-based risk scoring is meaningless for new junctions.
- **Fix:** Load `historical_coords.csv` at inference and compute real frequency lookups.

### 2.4 Diversion Engine Only Covers 7 Hardcoded Junctions
- **File:** `src/diversion_engine.py:9-52`
- **Problem:** `PREDEFINED_DIVERSIONS` has exactly 7 Bengaluru junctions. All other junctions fall into generic dynamic text that names nearest BallTree neighbors — no actual route knowledge.
- **Impact:** For 95%+ of real incidents, diversion "plan" is a generic template with junction names inserted. Not operationally useful.
- **Fix:** Either expand the predefined set (requires domain expert input), or integrate with a routing API (OSRM, Google Maps Directions) to generate real alternate paths.

### 2.5 Resource Recommender Has No Scaling for Event Size
- **File:** `src/resource_recommender.py:18-59`
- **Problem:** Personnel and barricade ranges are static per risk level regardless of event size, crowd count, affected road width, or number of lanes. A High-risk festival with 50,000 attendees vs. a High-risk tree fall get identical recommendations.
- **Fix:** Add event-specific parameters (expected_crowd_size, num_lanes_blocked) to the recommender inputs.

---

## 3. Data Quality Issues

### 3.1 Coordinate Imputation Silently Masks Bad Location Data
- **File:** `src/preprocessing.py:61-80`
- **Problem:** Invalid coordinates (outside Bengaluru bounds) are replaced with police station mean, then zone mean, then global median. No flag is added to indicate a row has imputed coordinates.
- **Impact:** Map visualizations in Tab 1 show events clustered at zone centroids, creating false hotspot patterns.
- **Fix:** Add `coord_imputed: bool` flag column. Filter imputed coords out of heatmap/scatter map or render them differently.

### 3.2 Duration Imputation Creates Training Signal Leakage
- **File:** `src/preprocessing.py:98-110`
- **Problem:** Missing durations filled from `groupby(['event_cause', 'priority']).median()`. This is done before train/test split, so test set durations are derived from training set statistics.
- **Fix:** Same as 1.3 — compute imputation statistics on training split only.

### 3.3 `requires_road_closure` Mapping Drops Unrecognized Values Silently
- **File:** `src/preprocessing.py:38-43`
- **Problem:** `.map({...}).fillna(False)` — any unrecognized string (e.g., `'Yes'` with capital Y, `'T'`, `'1.0'`) becomes `False` without logging.
- **Fix:** Log a warning with count of unrecognized values before `fillna`.

### 3.4 `min()` for End Timestamp Picks Earliest, Not Best
- **File:** `src/preprocessing.py:89`
- **Problem:** `valid_ends.min(axis=1)` picks the **earliest** end timestamp across `end_datetime`, `resolved_datetime`, `closed_datetime`. If `end_datetime` is `NaT` but `resolved_datetime` is valid, `min()` still works — but if a column has an erroneous early date, it silently uses that, producing negative or near-zero durations that get capped.
- **Fix:** Filter per-row: ignore timestamps that precede `start_datetime`, then take earliest valid one.

---

## 4. App / UX Issues

### 4.1 `load_data()` Runs Heavy Preprocessing on First Load
- **File:** `app.py:111-132`
- **Problem:** If processed CSV not found, `preprocess_data()` + `engineer_features()` + `calculate_risk_score()` run synchronously on Streamlit load. BallTree over full dataset built in main thread. Can take 30-60s with no progress indicator.
- **Fix:** Add `st.spinner()` during fallback. Better: always require `run_pipeline.py` to be run first; show clear error if processed data absent.

### 4.2 `diversion_engine` Initialized with `df_all` (Full Event Data), Not `coords_df`
- **File:** `app.py:143`
- **Problem:** `get_diversion_engine(df_all)` passes the full events DataFrame (8k+ rows) to `DiversionEngine`, which then builds BallTree over all event coordinates (many duplicates per junction). Should use `coords_df` (deduplicated).
- **Fix:** `get_diversion_engine(coords_df)`.

### 4.3 No Input Validation on Forecast Form Coordinates
- **File:** `app.py:473-474`
- **Problem:** User can enter any latitude/longitude. No bounds check for Bengaluru region. Out-of-bounds coords silently get density = 0 and arbitrary diversion junctions.
- **Fix:** Add validation: warn if lat/lon outside `[12.5–13.5, 77.2–77.9]`.

### 4.4 Sidebar Filters Applied Only Inside `if df_all is not None` Block
- **File:** `app.py:150-175`
- **Problem:** If `df_all` is None, `filtered_df` is never defined. Tab 1 code references `filtered_df` unconditionally inside `else` block (line 183+). Will raise `NameError` if data partially loads.
- **Fix:** Initialize `filtered_df = pd.DataFrame()` before the conditional.

---

## 5. Code Maintainability Issues

### 5.1 Hardcoded Absolute File Paths in Source Files
- **Files:** `src/preprocessing.py:129`, `src/feature_engineering.py:69`, `run_pipeline.py:22-24`
- **Problem:** Paths like `c:\Users\Harshitha\Desktop\Grid lock project\...` embedded in source. Will break on any other machine or if project moved.
- **Fix:** Use `pathlib.Path(__file__).parent.parent` relative paths or an `.env` / config file.

### 5.2 `recommend_resources_df()` Uses `iterrows()` on Full Dataset
- **File:** `src/resource_recommender.py:114`
- **Problem:** Row-by-row Python loop. For 8k rows, 8000 function calls in Python. Slow.
- **Fix:** Vectorize using `df.apply()` or map the deterministic outputs directly via lookup tables.

### 5.3 `calculate_risk_score()` Uses `iterrows()` for Cause Scores
- **File:** `src/risk_engine.py:38-59`
- **Problem:** Same issue — Python loop to compute cause scores. Easily vectorizable.
- **Fix:** Use `df['event_cause'].map(cause_score_dict)` with a dict lookup.

### 5.4 No Version Pinning for Models vs. Data
- **File:** `models/`
- **Problem:** Saved `.joblib` models have no version metadata. If dataset schema changes and pipeline is re-run, old models in `models/` are silently overwritten. No check that `app.py` model version matches pipeline version.
- **Fix:** Save model metadata (training date, feature list hash, dataset row count) alongside models. Validate on load in `predict_event()`.

---

## 6. Missing Capabilities (Problem Statement Gaps)

| Gap | Problem Statement Requirement | Current State |
|-----|-------------------------------|---------------|
| No event forecasting for future planned events | "forecast event-related traffic impact" | Only scores events already in dataset or manually entered |
| No post-event learning | "No post-event learning system" (stated problem) | No feedback loop — outcomes never recorded |
| No external data integration | Real-time feeds (traffic sensors, weather, CCTV) | Entirely offline batch system |
| No alert / notification system | "recommend optimal manpower" proactively | Passive dashboard only, no push alerts |
| No historical trend comparison | "historical data used to forecast" | Frequency counts only, no time-series modeling |

---

## Summary Priority Table

| # | Issue | Severity | Effort |
|---|-------|----------|--------|
| 1.1 | Duration regressor R²=0.21 | Critical | High |
| 1.2 | Classifier trained on its own labels | Critical | High |
| 1.3 | Data leakage in feature engineering | High | Medium |
| 2.1 | Scoring formula duplicated 3× | High | Low |
| 2.2 | ML level vs formula score disagreement | High | Low |
| 2.4 | Diversion only covers 7 junctions | High | High |
| 3.1 | Imputed coords create false hotspots | High | Low |
| 3.4 | Duration min() picks wrong timestamp | Medium | Low |
| 4.2 | DiversionEngine initialized with full df | Medium | Low |
| 5.1 | Hardcoded absolute paths | Medium | Low |
| 1.3 | Duration leakage into test set | High | Medium |
| 5.2/5.3 | iterrows() performance | Low | Low |
