# Road Accident Data Mining

*Data mining of 1M+ UK road-safety records — EDA, Apriori association rules, KMeans clustering, and a fatal-injury classifier.*

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![pandas](https://img.shields.io/badge/pandas-data--analysis-150458) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-f7931e) ![seaborn](https://img.shields.io/badge/seaborn-visualization-4c72b0) ![License: MIT](https://img.shields.io/badge/License-MIT-green)

This project (module 771762 — Big Data and Data Mining) analyzes UK road traffic accident data to explore when and where accidents occur, mine associations between contributing factors, cluster accidents geographically, detect outliers, and build a classification model to predict fatal injuries — with the aim of informing road-safety measures.

## Project Structure

```
Road-Accident-Data-Mining/
├── README.md
├── LICENSE
├── requirements.txt
├── traffic_accident_analysis.ipynb     # main notebook
├── Data/
│   └── accident_data_v1.0.0_2023.db    # SQLite database (add this — not committed)
├── Figures/                            # plots (embedded below)
├── Tables/                             # summary tables (CSV + Markdown)
├── Results/                            # analytical outputs (rules, clusters, model metrics)
└── Scripts/
    └── generate_results.py             # reproducible results pipeline
```

> **Note:** The notebook reads from a SQLite database at `Data/accident_data_v1.0.0_2023.db` containing the `accident`, `vehicle`, `casualty`, and `lsoa` tables. The database is ~200 MB — larger than GitHub's 100 MB file limit — so it is **not** committed to the repo (it is gitignored). Download it from the UK road-safety open data and place it in the `Data/` folder before running.

## What the Notebook Does

1. **Data assembly** — connects to the SQLite database, inspects each table, and builds pandas DataFrames for accident year 2020.
2. **Cleaning** — cleans the accident, vehicle, and casualty tables (types, missing values, joins on LSOA).
3. **Analysis** — temporal patterns for all accidents, motorbikes, and pedestrians; association-rule mining with the Apriori algorithm; geographic clustering for the Kingston-upon-Hull / Humberside / East Riding region.
4. **Outlier detection** — flags unusual entries and assesses whether to keep them.
5. **Modelling** — a classification model to predict fatal injuries, with class balancing (SMOTE / undersampling), feature-importance analysis, and evaluation via accuracy, precision, recall, F1, and a confusion matrix.

## Requirements

- Python 3.9+
- Packages in `requirements.txt`: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `mlxtend`, `folium`, `imbalanced-learn`, `notebook` (`sqlite3` and `itertools` are part of the standard library).

## Setup and Run

### Option 1 — Locally with Jupyter

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch Jupyter
jupyter notebook
```

Then open `traffic_accident_analysis.ipynb` and run the cells top to bottom.

### Option 2 — Google Colab

1. Upload `traffic_accident_analysis.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Upload `accident_data_v1.0.0_2023.db` into a `Data/` folder (or mount Google Drive).
3. Install the extra packages if needed: `!pip install mlxtend folium imbalanced-learn`
4. Run each cell in order.

## Analysis & Results

### 1. Temporal patterns — all accidents

Accidents peak during commuting hours (a smaller morning peak around 08:00 and a larger evening peak around 17:00), and are most frequent on Fridays, dropping notably on weekends.

![Accidents by hour](Figures/01_accidents_by_hour.png)
![Accidents by day](Figures/02_accidents_by_day.png)

### 2. Motorbike accidents by engine class

Broken down for motorcycles ≤125cc, 125–500cc, and 500cc+ — by day of week and hour of day. Smaller bikes track the commuting pattern, while larger bikes (500cc+) show a distinct weekend/leisure peak on Sundays.

![125cc by day](Figures/03_motorcycle_125cc_by_day.png)
![125cc by hour](Figures/04_motorcycle_125cc_by_hour.png)
![125–500cc by day](Figures/05_motorcycle_125_500cc_by_day.png)
![125–500cc by hour](Figures/06_motorcycle_125_500cc_by_hour.png)
![500cc+ by day](Figures/07_motorcycle_500cc_plus_by_day.png)
![500cc+ by hour](Figures/08_motorcycle_500cc_plus_by_hour.png)

### 3. Pedestrian accidents

Pedestrian involvement peaks in the afternoon (around 15:00–16:00, aligned with the school/commute window) and is highest on Fridays.

![Pedestrian by day](Figures/09_pedestrian_by_day.png)
![Pedestrian by hour](Figures/10_pedestrian_by_hour.png)

### 4. Regional clustering

Using the elbow method to select the number of clusters, then clustering accidents in the Hull / Humberside / East Riding region to reveal their geographic distribution.

![Elbow method](Figures/11_clustering_elbow.png)
![Regional clusters](Figures/12_regional_clusters_map.png)

Reproduced as static maps from the database: accident locations across the Humberside region, and the resulting KMeans clusters (k=5) with centroids. The dense central cluster (Cluster 3) corresponds to Kingston upon Hull, with outer clusters covering the surrounding East Riding towns.

![Regional accidents map](Figures/26_region_accidents_map.png)
![Regional KMeans clusters](Figures/27_region_clusters_kmeans.png)

**Cluster centroids** (from `Results/cluster_centroids.csv`):

| cluster | longitude | latitude | n_accidents |
|--------:|----------:|---------:|------------:|
| 1 | -0.826 | 53.810 | 128 |
| 2 | -0.271 | 54.045 | 109 |
| 3 | -0.362 | 53.759 | 869 |
| 4 | -0.646 | 53.592 | 267 |
| 5 | -0.092 | 53.576 | 336 |

Per-accident cluster assignments are saved in `Results/cluster_assignments.csv` (1,709 rows).

### 5. Fatal-injury classification

The target class is highly imbalanced (fatal injuries are rare), so class balancing (SMOTE / undersampling) is applied before modelling. Feature-importance analysis highlights `casualty_severity`, `speed_limit`, and `urban_or_rural_area` as the strongest predictors, and the model achieves **96.22% accuracy** on the balanced test set.

| metric | value |
|:-------|------:|
| Accuracy | 0.962 |
| Precision | 0.970 |
| Recall | 0.954 |
| F1 | 0.962 |

![Class distribution](Figures/13_class_distribution.png)
![Feature importance](Figures/14_feature_importance.png)
![Confusion matrix](Figures/15_confusion_matrix.png)

## Extended Data Exploration

An additional exploration of the full 2020 accident records (91,199 accidents, 600,332 casualties, 849,091 vehicles), with summary tables and figures generated directly from the database. Codes are decoded using the DfT road-safety data guide.

### Severity, timing and conditions

Most accidents are slight; fatal accidents are rare (1.5%). Accidents are dominated by fine weather and daylight — not because those conditions are dangerous, but because that is when most driving happens.

**Accident severity (2020)**

| severity | count  | pct  |
|:---------|-------:|-----:|
| Fatal    |  1,391 |  1.5 |
| Serious  | 18,355 | 20.1 |
| Slight   | 71,453 | 78.3 |

**Accidents by weather condition**

| weather         | count  | pct  |
|:----------------|-------:|-----:|
| Fine            | 70,729 | 77.6 |
| Raining         | 11,583 | 12.7 |
| Other           |  2,629 |  2.9 |
| Unknown         |  2,423 |  2.7 |
| Raining + winds |  1,665 |  1.8 |
| Fine + winds    |  1,401 |  1.5 |
| Fog/mist        |    510 |  0.6 |
| Snowing         |    185 |  0.2 |
| Snowing + winds |     73 |  0.1 |

![Severity distribution](Figures/16_severity_distribution.png)
![Accidents by month](Figures/17_accidents_by_month.png)
![Severity by day](Figures/18_severity_by_day.png)
![Weather conditions](Figures/19_weather_conditions.png)
![Light conditions](Figures/20_light_conditions.png)
![Road surface](Figures/21_road_surface.png)

### Road environment and severity

Speed limit and urban/rural setting relate strongly to severity: most accidents happen on 30 mph urban roads, but fatal accidents are disproportionately represented at 60 mph (rural roads).

**Speed limit vs severity**

| speed_limit | Fatal | Serious | Slight |
|------------:|------:|--------:|-------:|
| 20          |    53 |   1,707 |  9,423 |
| 30          |   468 |   9,842 | 41,950 |
| 40          |   133 |   1,725 |  6,009 |
| 50          |   116 |     821 |  2,845 |
| 60          |   476 |   3,251 |  7,681 |
| 70          |   145 |   1,006 |  3,536 |

![Speed limit](Figures/22_speed_limit.png)
![Urban vs rural severity](Figures/23_urban_rural_severity.png)
![Road type](Figures/24_road_type.png)

### Who is involved

Pedestrians make up a small share of casualties but a disproportionate share of fatalities. Cars account for roughly two-thirds of vehicles involved.

**Casualties by class and severity (2020)**

| class        | Fatal | Serious | Slight |
|:-------------|------:|--------:|-------:|
| Driver/Rider |   916 |  13,519 | 64,895 |
| Passenger    |   198 |   2,678 | 18,628 |
| Pedestrian   |   346 |   3,905 | 10,499 |

**Top vehicle types (2020)**

| Vehicle           | count   | pct  |
|:------------------|--------:|-----:|
| Car               | 114,145 | 68.2 |
| Pedal cycle       |  16,766 | 10.0 |
| Van/Goods ≤3.5t   |  10,338 |  6.2 |
| M/cycle 125cc-    |   7,523 |  4.5 |
| M/cycle 500cc+    |   3,784 |  2.3 |
| Taxi/Private hire |   2,612 |  1.6 |
| Goods 7.5t+       |   2,501 |  1.5 |
| Bus/Coach         |   2,213 |  1.3 |
| M/cycle 125-500cc |   1,691 |  1.0 |
| Other             |   1,185 |  0.7 |

### Geographic distribution

Accidents cluster around major population centres (London, the Midlands, the North West), tracing the shape of Great Britain's road network. Fatal accidents (dark red) are more scattered and rural.

![Geographic distribution by severity](Figures/25_geographic_severity.png)

All extended tables are also saved as CSV and Markdown in the `Tables/` folder.

## Reproducible Results

The script `Scripts/generate_results.py` regenerates the analytical outputs directly from the database into the `Results/` folder (association rules, cluster data, and model evaluation), using only numpy/pandas. Run it with `python Scripts/generate_results.py`.

### Association-rule mining (Apriori)

Mining accident conditions surfaces intuitive but useful associations — for example, rain co-occurs with wet road surfaces in urban 30 mph zones (lift ≈ 3.5), and 60 mph roads strongly associate with rural settings (lift ≈ 3.0), confirming the data's internal consistency. All 25 rules (min support 0.05, lift > 1):

| antecedent | consequent | support | confidence | lift |
|:-----------|:-----------|--------:|-----------:|-----:|
| weather=Raining | surface=Wet, urban_rural=Urban | 0.082 | 0.641 | 3.53 |
| surface=Wet, urban_rural=Urban | weather=Raining | 0.082 | 0.452 | 3.53 |
| weather=Raining | speed=30mph, surface=Wet | 0.068 | 0.534 | 3.42 |
| speed=30mph, surface=Wet | weather=Raining | 0.068 | 0.438 | 3.42 |
| weather=Raining | severity=Slight, surface=Wet | 0.099 | 0.775 | 3.41 |
| severity=Slight, surface=Wet | weather=Raining | 0.099 | 0.437 | 3.41 |
| weather=Raining | light=Daylight, surface=Wet | 0.069 | 0.536 | 3.36 |
| light=Daylight, surface=Wet | weather=Raining | 0.069 | 0.431 | 3.36 |
| speed=30mph, weather=Raining | surface=Wet | 0.068 | 0.972 | 3.34 |
| surface=Wet | speed=30mph, weather=Raining | 0.068 | 0.234 | 3.34 |
| urban_rural=Urban, weather=Raining | surface=Wet | 0.082 | 0.969 | 3.33 |
| surface=Wet | urban_rural=Urban, weather=Raining | 0.082 | 0.282 | 3.33 |
| weather=Raining | surface=Wet | 0.124 | 0.968 | 3.32 |
| severity=Slight, weather=Raining | surface=Wet | 0.099 | 0.968 | 3.32 |
| surface=Wet | weather=Raining | 0.124 | 0.425 | 3.32 |
| surface=Wet | severity=Slight, weather=Raining | 0.099 | 0.340 | 3.32 |
| light=Daylight, weather=Raining | surface=Wet | 0.069 | 0.964 | 3.31 |
| surface=Wet | light=Daylight, weather=Raining | 0.069 | 0.235 | 3.31 |
| light=Daylight, speed=60mph | urban_rural=Rural | 0.088 | 0.967 | 2.97 |
| urban_rural=Rural | light=Daylight, speed=60mph | 0.088 | 0.271 | 2.97 |
| speed=60mph | light=Daylight, urban_rural=Rural | 0.088 | 0.698 | 2.97 |
| light=Daylight, urban_rural=Rural | speed=60mph | 0.088 | 0.375 | 2.97 |
| speed=60mph | urban_rural=Rural | 0.122 | 0.965 | 2.96 |
| urban_rural=Rural | speed=60mph | 0.122 | 0.374 | 2.96 |
| speed=60mph, surface=Dry | urban_rural=Rural | 0.073 | 0.965 | 2.96 |

Also saved as `Results/association_rules.csv`.

### Fatal-injury classifier — leakage-free cross-check

As a robustness check, `generate_results.py` also fits a balanced logistic-regression model using **only pre-outcome features** (speed limit, vehicles, casualties, urban/rural, light, road surface, road type, day) — deliberately excluding casualty severity. On this harder, leakage-free setup it reaches F1 ≈ 0.61, which reflects how difficult predicting rare fatal outcomes is from circumstances alone.

| metric | value |
|:-------|------:|
| Accuracy | 0.67 |
| Precision | 0.66 |
| Recall | 0.58 |
| F1 | 0.61 |

![Confusion matrix](Results/confusion_matrix.png)

## Key Findings

- Accidents concentrate in commuting hours and peak on Fridays; weekends are quieter overall.
- Larger motorbikes (500cc+) buck the commuter trend with a strong Sunday/leisure peak.
- Pedestrian accidents peak mid-to-late afternoon and account for a disproportionate share of fatalities.
- Severity is highest on high-speed rural roads: 60 mph roads produce far more fatalities per accident than 30 mph urban roads.
- Fatal outcomes are rare (1.5%), so the classification model requires class balancing; severity, speed limit, and urban/rural setting are the most informative features.

## About

This project began as a university assignment (module 771762, Big Data and Data Mining) and was reworked into a standalone portfolio project. The dataset is derived from publicly available UK road-safety open data.

## License

Released under the [MIT License](LICENSE).
