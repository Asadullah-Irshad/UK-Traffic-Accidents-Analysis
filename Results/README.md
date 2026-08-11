# Results

Analytical outputs generated from the 2020 accident data by `Scripts/generate_results.py`.

| File | Description |
|:-----|:------------|
| `association_rules.csv` / `.md` | Top association rules from Apriori mining of accident conditions (support, confidence, lift). |
| `cluster_assignments.csv` | Each Humberside-region accident with its KMeans cluster (1–5) and coordinates. |
| `cluster_centroids.csv` | The five cluster centres with their accident counts. |
| `model_metrics.csv` / `.json` | Evaluation of the fatal-injury classifier (accuracy, precision, recall, F1, confusion matrix). |
| `confusion_matrix.png` | Confusion matrix for the fatal-injury classifier. |

To regenerate, place the database in `../Data/` (or set `ACC_DB`) and run:

```bash
python Scripts/generate_results.py
```
