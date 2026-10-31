# NariSafe: Women Safety Risk Awareness

**She Code Africa ML/AI Challenge 2026 submission**

NariSafe is a supervised-learning (multi-class classification) project that predicts the **safety risk level (`low` / `medium` / `high`)** for women in public spaces across 34 Indian cities. It combines official crime statistics with environmental context such as time of day, area type, street lighting, crowd density and police access.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Repository Structure](#2-repository-structure)
3. [The Dataset](#3-the-dataset)
4. [Key EDA Findings](#4-key-eda-findings)
5. [Methodology](#5-methodology)
6. [Results](#6-results)
7. [Explainability & Model Insights](#7-explainability--model-insights)
8. [Innovation in Approach](#8-innovation-in-approach)
9. [How to Run](#9-how-to-run)

---

## 1. Problem Statement

Women in India face very different levels of risk depending on **where** and **when** they are in public. Risk tends to rise in transit and industrial areas, and late at night when streets are poorly lit and few people are around.

**Goal:** understand what drives high-risk situations and build a model that predicts `risk_level` from time, place, lighting, police access and city-level crime context.

Because only about 5% of rows are `high` risk, and missing a high-risk situation is costlier than a false alarm, we optimise for **macro F1** and **recall on the `high` class**, not plain accuracy.

---

## 2. Repository Structure

```
.
├── narisafe_eda_model_final.ipynb      # Full EDA + modelling notebook
├── narisafe_risk_pipeline.joblib       # Saved preprocessing + XGBoost pipeline
├── narisafe_model_metadata.json        # Features, params, metrics, locked test cities
├── women_risk_awareness_-_narisafe_ml_features.csv   # Provided dataset (add if permitted)
└── README.md
```

---

## 3. The Dataset

The dataset was provided by the challenge; no external data was used.

| Property | Value |
| --- | --- |
| Rows × columns | 489,804 × 40 |
| Cities | 34 |
| Complaint (crime) types | 42 |
| Area contexts | 7 (e.g. commercial, market, residential, educational, mixed, industrial, transit) |
| Target | `risk_level` (low 36.3% · medium 58.8% · high 5.0%) |
| Missing values | None |

**Column groups**

- **Time & place:** `city`, `day_of_week`, `hour`, `time_bucket`, `is_weekend`, `area_context`, `crowd_density`, `latitude`, `longitude`
- **Crime statistics (NCRB-style):** `women_crime_2021/22/23`, `women_crime_rate_2023`, `chargesheeting_rate_2023`, `women_crime_growth_21_23`, `complaint_type_clean`, `complaint_type_count`, `complaint_type_share`, `complaint_severity`
- **Infrastructure:** police stations, nearest police station distance, transport/bus/rail counts, street lights, road length/density, land-use counts (commercial, residential, industrial, retail, education)
- **Engineered scores:** `lighting_score`, `police_access_score`, `transport_access_score`, `urban_density_score`

---

## 4. Key EDA Findings

1. **The data is a complete grid, not a log of incidents.** Rows are the full cross-product of 34 cities × 7 days × 7 hours × 7 area types × 42 crime types (every city has exactly 14,406 rows). 28 of 40 columns are constant within a city, so there are effectively **34 city profiles**.
2. **Time of day is the strongest pattern.** High-risk share is 20.8% late at night, 6.4% in the evening, 0.4% in the morning and 0.1% in the afternoon. Average risk peaks at 11 PM.
3. **Weekday has no effect.** Every day (and weekend vs weekday) has an identical 5.0% high-risk share.
4. **Area and crowd matter.** Industrial (16.1%) and transit (8.4%) areas carry the most high-risk labels; commercial areas the fewest (1.9%). Low crowd density has a 12.8% high share vs 1.3% for crowded areas.
5. **Severity matters.** High-risk share rises with complaint severity: 2.0% → 5.4% → 8.6% for severity 1 → 2 → 3.
6. **City differences are large.** High share ranges from 0.2% (Madurai) to 25.6% (Vasai Virar). Higher-risk rows come from cities with higher crime rates, longer distances to police stations and fewer commercial/education places.
7. **Lighting and police access separate the classes.** Mean `lighting_score` falls from 3.43 (low) to 2.29 (high); `police_access_score` from 2.52 to 1.47.
8. **Data quality notes.** Crime counts are heavily right-skewed (median 0), twelve column pairs correlate above 0.8, and real street-light data exists for only 3 cities.
9. **Trends.** Between 2021 and 2023 reported crime rose in 22 cities and fell in 12 (median rise 9%). A fall in registered cases can reflect reporting changes as much as real safety.

---

## 5. Methodology

### Step 1: Diagnose the data before modelling
Seven structural checks were run on the data:

- Every distinct feature vector maps to exactly **one** label, so `risk_level` is computed from the features rather than observed.
- Weekday changes nothing: 489,804 rows collapse to **69,972 distinct feature vectors** (exactly 7×). Training uses these de-duplicated rows.
- `time_bucket` is a function of `hour`; `crowd_density` is a lookup of area and hour; `police_access_score` is a function of `nearest_police_station_km`.

### Step 2: Choose an honest validation strategy
A random row split gives a perfect **1.000** on every metric, because identical rows and the same city appear in both train and test, and `city` / coordinates let the model identify the city. With the same model and features but **whole cities held out** (`GroupKFold` by city), accuracy drops to 0.886 and recall on `high` to 0.63. Every later number therefore uses city-grouped validation.

### Step 3: Feature selection by ablation
Information was added in six layers and each was scored with grouped CV. Identifiers (`city`, coordinates, `complaint_type_clean`), weekday columns and raw yearly counts were excluded from all layers. The **one-standard-error rule** picked the smallest layer within one SE of the best.

| Layer | Features | Macro F1 | Recall (high) |
| --- | --- | --- | --- |
| Context only | 5 | 0.621 | 0.295 |
| + complaint count | 6 | 0.697 | 0.422 |
| + crime rate | 7 | 0.771 | 0.585 |
| + chargesheeting & growth | 9 | 0.798 | 0.613 |
| **+ police stations (selected)** | **11** | **0.847** | **0.635** |
| All city-level columns | 28 | 0.857 | 0.657 |

**Final 11 features:** `hour`, `area_context`, `crowd_density`, `lighting_score`, `complaint_severity`, `complaint_type_count`, `women_crime_rate_2023`, `chargesheeting_rate_2023`, `women_crime_growth_21_23`, `police_station_count_5km`, `nearest_police_station_km`.

### Step 4: Model & tuning
- **Model:** XGBoost multi-class classifier (`multi:softprob`) inside a scikit-learn `Pipeline` with ordinal encoding for `area_context` and `crowd_density`.
- **Tuning:** shallow, regularised trees with early stopping, validated by grouped 5-fold CV. Depth 5 reached lower training loss but worse validation loss (memorising cities), so **depth 3, `min_child_weight=20`, 237 rounds, `eta=0.1`, `reg_lambda=5`** was chosen.
- **Locked test set:** 7 cities (Faridabad, Kota, Meerut, Prayagraj, Ranchi, Thiruvananthapuram, Thrissur) were held out and never used for any decision.

### Step 5: Class imbalance
Three schemes were compared on grouped out-of-fold predictions:

| Weighting | Macro F1 | Recall (high) |
| --- | --- | --- |
| None | 0.821 | 0.567 |
| Square-root | 0.845 | 0.679 |
| **Balanced (chosen)** | **0.855** | **0.824** |

Balanced weights were chosen because missing a high-risk situation is worse than a false alarm.

### Step 6: Final training & export
The final pipeline is retrained on all de-duplicated rows and saved with joblib, along with a metadata JSON file.

---

## 6. Results

| Evaluation | Accuracy | Balanced Acc. | Macro F1 | Recall (high) | QWK |
| --- | --- | --- | --- | --- | --- |
| A. Random split (memorisation, not trustworthy) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B. Cities held out, untuned baseline | 0.886 | 0.806 | 0.833 | 0.630 | 0.812 |
| C. Final model, in-sample (seen cities) | 0.975 | 0.985 | 0.938 | 1.000 | n/a |
| **D. Final model, grouped CV (27 cities)** | **0.889** | **0.873** | **0.855** | **0.824** | **0.831** |
| **E. Final model, locked test (7 cities)** | **0.923** | **0.941** | **0.868** | **0.957** | n/a |

**Reading the table**

- **D is the most reliable estimate** for new cities, since it averages over all 27 training cities. E is a single draw of 7 cities, so its higher numbers should not be treated as the expected performance.
- Versus the baseline (B), the final model keeps similar accuracy but lifts macro F1 from 0.833 to 0.855 and recall on `high` from 0.63 to 0.824.
- On the locked test, `high` recall is 0.957 but precision is 0.590. The model catches nearly all high-risk rows at the cost of extra false alarms, which is the deliberate trade-off of balanced weights.
- The hardest cities to predict when held out are Rajkot, Dhanbad, Vasai Virar, Aurangabad and Vishakhapatnam.

---

## 7. Explainability & Model Insights

Permutation importance was computed on the **unseen locked-test cities** (drop in macro F1 when a feature is shuffled):

| Feature | Importance |
| --- | --- |
| `lighting_score` | 0.190 |
| `women_crime_rate_2023` | 0.150 |
| `hour` | 0.148 |
| `complaint_severity` | 0.120 |
| `complaint_type_count` | 0.106 |
| `crowd_density` | 0.069 |
| `women_crime_growth_21_23` | 0.017 |
| `nearest_police_station_km` | 0.003 |
| `area_context` | 0.002 |
| `police_station_count_5km` | 0.001 |
| `chargesheeting_rate_2023` | ~0 |

**Takeaways**

- The model relies on two sources of signal: the **time-and-place pattern** (lighting, hour, crowd) and **how serious the city and crime are** (crime rate, severity, count).
- `lighting_score`, `crowd_density`, `hour` and `area_context` are all derived from the same area-and-hour grid, so they share credit and each looks smaller than the group does together.
- Police features helped cross-validated scores but matter little on the 7 test cities, which suggests city-level effects are hard to pin down with only 34 cities.
- The notebook also includes confusion matrices and a per-city accuracy chart showing which cities are hard to predict from the others.

---

## 8. Innovation in Approach

- **Data forensics first:** formal checks proved the file is a complete grid and that the label is a deterministic function of the features, which changed how we evaluated everything.
- **Leakage-aware evaluation:** we show side by side how a random split gives a "perfect" score and how a city-held-out split gives the honest one.
- **Principled feature selection:** layered ablation with the one-standard-error rule, instead of using every column.
- **Overfitting control:** depth, `min_child_weight` and early stopping chosen on grouped CV, with a locked test of 7 cities.
- **Safety-aware metric choice:** macro F1, `high` recall and quadratic weighted kappa (which respects the ordered low < medium < high scale) alongside accuracy.
- **Deployable artefact:** a single serialised pipeline plus metadata JSON.

---

## 9. How to Run

1. **Clone the repository and open file in google colab**
    * If choosing to run in a code editor, cd into repository
2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
   ```
3. **Add the dataset** (`women_risk_awareness_-_narisafe_ml_features.csv`) to the project root or update the `FILE` path in the notebook's loading cell.
4. **Run the notebook**
   ```bash
   jupyter notebook narisafe_eda_model.ipynb
   ```
   Or upload it to Google Colab and run all cells as it m approach runs on Google Colab. The random seed is fixed (`SEED = 42`) for reproducibility. Training takes under a couple of minutes on a standard CPU.
---
*Built for the She Code Africa ML/AI Challenge 2026.*
