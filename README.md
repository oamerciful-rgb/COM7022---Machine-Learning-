# COM7022 Machine Learning – Product Segmentation with Clustering

Clustering analysis of 1,000 men's fragrance listings from an e-commerce catalogue.
Products are grouped by **price**, **units sold** and **available stock** using
**K-Means**, **Hierarchical clustering (Ward)** and **DBSCAN**, and the resulting
segments are interpreted for business use.

## Repository contents

| File | Purpose |
|---|---|
| `COM7022-Part1_Data_Preparation.ipynb` | Loads the raw data, recovers missing values, corrects errors, standardises categories and applies log transformations. Saves `sales_clean.csv`. |
| `COM7022-Part2_Clustering.ipynb` | Feature selection, scaling, PCA, K-Means, Hierarchical clustering, DBSCAN, model comparison and segment profiles. Saves `sales_clustered.csv`. |
| `COM7022-Part3_Robustness_and_Figures.ipynb` | IQR outlier analysis, outlier-treatment comparison, silhouette analysis, subsampling and bootstrap stability, tests of product size and update history as features, and the report figures (saved to `figures/`). |
| `requirements.txt` | Python packages needed to run the notebooks. |

## Data

The dataset (`Sales-[5444].csv`) was provided by Arden University for assessment
and is **not included** in this repository. To reproduce the results, place your
copy of the file in the same folder as the notebooks (any file name containing
`5444` is detected automatically).

## How to reproduce the results

**Option A – Google Colab (no installation)**
1. Open a notebook from this repository and click **Open in Colab**.
2. Upload the required file using the folder icon on the left:
   - Part 1: the raw dataset (`Sales-[5444].csv`)
   - Part 2 and Part 3: `sales_clean.csv` (created by Part 1). Part 3 also uses the raw dataset for the missing-values chart.
3. Click **Runtime → Run all**.

**Option B – Local Python**
```bash
pip install -r requirements.txt
jupyter notebook
```
Run the notebooks **in order: Part 1 → Part 2 → Part 3**, with the dataset in the same folder.

All models use `random_state = 42`, so results are reproducible.

## Main results

| Model | Clusters | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---|---|---|---|---|
| **K-Means (k = 4) – selected** | 4 | 0.300 | 1.028 | 493.6 |
| Hierarchical, Ward (k = 4) | 4 | 0.250 | 1.166 | 386.5 |
| DBSCAN (eps = 0.6, min_samples = 8)* | 3 | 0.279 | 0.685 | 58.5 |

\*Scores exclude the 4.9% of products labelled as noise; one cluster contains about 92% of products.

**Stability of the K-Means segments:** mean Adjusted Rand Index of 0.996 across 10 random seeds,
0.94 across 50 subsamples (80% of products) and 0.87 across 50 bootstrap samples.

**Segments identified:**

| Segment | Share of products | Median price | Median units sold | Share of estimated revenue |
|---|---|---|---|---|
| Bestsellers | 26% | $30.00 | 803 | 87% |
| Premium niche | 27% | $68.60 | 35 | 8% |
| Budget range | 24% | $19.00 | 31 | 1% |
| Low-stock slow movers | 23% | $40.00 | 14 | 4% |

## Author

Anthony Merciful Okwori – MSc Data Science, Arden University
