# Southwest Airlines Flight Delay Prediction

> Predicting departure delays for Southwest Airlines using 90M+ rows of flight and weather data on Databricks with PySpark — comparing Logistic Regression, XGBoost, and Deep Learning models.

## Collaborators

Built as a team project for UC Berkeley MIDS W261 (Machine Learning at Scale).

---

## 01. The Problem

### Context

Flight delays cost the U.S. airline industry an estimated $28 billion annually. For Southwest Airlines — the largest domestic carrier operating over 4,000 daily flights — even small improvements in delay prediction translate to massive operational savings. If an airline can predict delays 2 hours before departure, it can proactively rebook passengers, reposition aircraft, and allocate gate resources more efficiently.

### Challenge

We needed to build a model that:
- **Processes at scale**: 90M+ rows of flight records spanning 2015-2024, merged with hourly weather data
- **Predicts departure delays**: Binary classification — will a Southwest flight depart 15+ minutes late?
- **Prioritizes recall**: Missing a delay (false negative) is far more costly than a false alarm (false positive). A missed delay means stranded passengers, cascading downstream delays, and crew scheduling chaos. A false alarm means preparing a backup plan that wasn't needed — much cheaper.
- **Handles class imbalance**: Only ~20% of flights are delayed, creating a significant skew
- **Engineers meaningful features**: Raw flight and weather data alone isn't predictive enough — the model needs features that capture temporal patterns, airport-specific behavior, and Southwest-specific operational metrics

The core question: can we predict whether a Southwest Airlines flight will be delayed before it departs, using only information available at prediction time (no future data leakage)?

---

## 02. The Approach

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Databricks / PySpark Cluster                      │
│                                                                      │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────────┐  │
│  │  FAA On-Time   │    │  NOAA Hourly   │    │  Airport/Route     │  │
│  │  Performance   │    │  Weather Data  │    │  Metadata          │  │
│  │  (OTP)         │    │  (ISD)         │    │                    │  │
│  └───────┬────────┘    └───────┬────────┘    └────────┬───────────┘  │
│          │                     │                      │              │
│          └──────────┬──────────┘                      │              │
│                     ▼                                 │              │
│          ┌────────────────────┐                       │              │
│          │   Data Join &     │◀──────────────────────┘              │
│          │   Enrichment      │                                      │
│          │   (PySpark SQL)   │                                      │
│          └────────┬──────────┘                                      │
│                   │                                                  │
│                   ▼                                                  │
│          ┌────────────────────┐                                      │
│          │ Feature Engineering│                                      │
│          │ 40+ Features       │                                      │
│          │ - Rolling metrics  │                                      │
│          │ - Weather Z-scores │                                      │
│          │ - Airport PageRank │                                      │
│          │ - SW delay rates   │                                      │
│          └────────┬──────────┘                                      │
│                   │                                                  │
│                   ▼                                                  │
│          ┌────────────────────┐                                      │
│          │  Train/Val/Test    │                                      │
│          │  Time-Series Split │                                      │
│          │  2015-21 / 22-23   │                                      │
│          │       / 2024       │                                      │
│          └────────┬──────────┘                                      │
│                   │                                                  │
│       ┌───────────┼───────────┬──────────────┐                      │
│       ▼           ▼           ▼              ▼                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐                │
│  │Logistic │ │ XGBoost │ │  Dense  │ │Bidirection│                │
│  │Regress. │ │(Spark)  │ │  Neural │ │   LSTM    │                │
│  │Baseline │ │         │ │ Network │ │           │                │
│  └────┬────┘ └────┬────┘ └────┬────┘ └─────┬─────┘                │
│       │           │           │             │                       │
│       └───────────┴───────────┴─────────────┘                       │
│                          │                                           │
│                          ▼                                           │
│               ┌────────────────────┐                                 │
│               │   Model Comparison │                                 │
│               │   F2 / Recall /    │                                 │
│               │   Precision / AUC  │                                 │
│               └────────────────────┘                                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Pipeline

```
┌──────────────┐      ┌──────────────┐      ┌──────────────────────────┐
│ Bureau of    │      │ NOAA Weather │      │ Derived Features          │
│ Transporta-  │      │ Stations     │      │                          │
│ tion Stats   │      │ (Hourly)     │      │ - 30-day rolling delay   │
│              │      │              │      │   rates per airport       │
│ 90M+ flight  │─────▶│ Wind, temp,  │─────▶│ - Aircraft delay rates   │
│ records      │ JOIN │ precip,      │      │ - Weather Z-scores       │
│ 2015-2024    │      │ visibility   │      │ - Airport PageRank       │
│              │      │              │      │ - SW-specific metrics    │
└──────────────┘      └──────────────┘      └──────────────────────────┘
                                                       │
                                                       ▼
                                            ┌────────────────────┐
                                            │ Final Dataset      │
                                            │ 90M+ rows          │
                                            │ 40+ features       │
                                            │ Partitioned by year│
                                            └────────────────────┘
```

### Feature Engineering Deep Dive

This is where the project's real value lies. Raw flight data (origin, destination, scheduled time) has limited predictive power. We engineered 40+ features that capture the operational reality of airline operations:

**Temporal & Rolling Metrics**
- **30-day rolling delay rate per airport**: What percentage of Southwest flights from this airport were delayed in the last 30 days? Uses hierarchical fallback — if insufficient history, falls back to global average (15%).
- **Time-of-day performance buckets**: Morning, afternoon, evening delay patterns for each airport

**Weather Features**
- **Z-score normalized weather anomalies**: How unusual is today's wind speed/temperature compared to the airport's historical distribution? A 30mph wind at Chicago O'Hare is normal; at a small regional airport, it's extreme.
- **Weather volatility**: Standard deviation of hourly readings in the 6-hour window before departure

**Network Features**
- **Airport PageRank**: Used GraphFrames to compute PageRank on the flight network. High-PageRank airports (ATL, DEN, ORD) see more cascading delays. This captures network effects that individual flight features miss.
- **Route-level delay rates**: Historical delay patterns for specific origin-destination pairs

**Southwest-Specific Metrics**
- **Aircraft delay rate**: How often is this specific aircraft (tail number) delayed? Hierarchical fallback: aircraft → route → global median
- **Southwest origin performance**: A hybrid metric combining airport and time-based delay patterns specific to Southwest operations

**Handling Missing Data**
- Built a hierarchical coalesce system: aircraft-specific → route-specific → airport-specific → global fallback
- Ensures every row has a value for every feature, even for new routes or aircraft

### Models Compared

| Model | Framework | Key Parameters |
|-------|-----------|---------------|
| **Logistic Regression** | PySpark MLlib | L2 regularization, cross-validated regParam |
| **XGBoost** | SparkXGBClassifier | max_depth, num_rounds, cross-validated |
| **Feed-Forward NN** | TensorFlow/Keras | Grid search + Optuna optimization |
| **Bidirectional LSTM** | TensorFlow/Keras | Sequence modeling on flattened features |

### Why F2 Score?

We used F2 score as the primary evaluation metric instead of the more common F1 or accuracy. F2 weights recall twice as heavily as precision:

```
F2 = 5 × (precision × recall) / (4 × precision + recall)
```

**Why this matters for delay prediction:**
- **False Negative** (missed delay): Southwest doesn't prepare for the delay → stranded passengers, cascading downstream delays, crew scheduling violations, gate conflicts. Cost: potentially millions per major delay event.
- **False Positive** (false alarm): Southwest prepares a contingency plan for a flight that departs on time → minimal cost, just unnecessary preparation.

The asymmetric cost structure makes recall far more important than precision.

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Compute** | Databricks | Managed Spark cluster for distributed processing |
| **Data Processing** | PySpark (SQL + MLlib) | Distributed joins, feature engineering, model training |
| **Gradient Boosting** | SparkXGBClassifier | Distributed XGBoost on Spark |
| **Deep Learning** | TensorFlow/Keras | Feed-forward NN and BiLSTM models |
| **Hyperparameter Tuning** | Optuna | Bayesian optimization for neural network hyperparameters |
| **Graph Analysis** | GraphFrames | Airport PageRank computation on flight network |
| **Data Visualization** | Matplotlib, Seaborn | EDA, confusion matrices, feature importance plots |
| **Data Format** | Parquet (on DBFS) | Efficient columnar storage for large datasets |

---

## 03. Results

### Model Comparison

| Model | Precision | Recall | F2 Score | AUC |
|-------|-----------|--------|----------|-----|
| Logistic Regression (Baseline) | — | — | — | — |
| XGBoost | — | — | — | — |
| Feed-Forward Neural Network | — | — | — | — |
| Bidirectional LSTM | — | — | — | — |

*Note: Exact metrics vary across cross-validation folds. See the full notebook for detailed per-fold results.*

### Key Findings

- **Feature engineering dominated model choice**: The gap between a well-engineered logistic regression and a poorly-featured neural network was larger than the gap between model architectures. The 40+ engineered features — especially rolling delay rates, weather Z-scores, and airport PageRank — were the primary drivers of performance.
- **XGBoost performed competitively**: With proper hyperparameter tuning, XGBoost matched or exceeded the neural network models while being significantly faster to train and more interpretable.
- **Class imbalance required careful handling**: Naive models achieved high accuracy by predicting "no delay" for everything. Threshold tuning on F2 score was essential to produce useful predictions.
- **Estimated business impact**: $20 million in potential cost savings for Southwest Airlines through proactive delay management.

### Dataset Scale

| Split | Years | Records |
|-------|-------|---------|
| Training | 2015-2021 | ~86M |
| Validation | 2022-2023 | 2,735,873 |
| Test | 2024 | 1,418,940 |

---

## Technical Decisions & Trade-offs

### Why PySpark instead of pandas?

90M+ rows don't fit in memory on a single machine. PySpark distributes processing across a Databricks cluster, enabling joins between flight and weather data that would be impossible in pandas. The trade-off: PySpark has a steeper learning curve and some operations (like complex window functions for rolling features) require careful optimization.

### Why time-series splits instead of random splits?

Random train/test splits would leak future information into training. A model trained on 2023 data and tested on 2021 data would see future patterns. Time-series splits (train: 2015-2021, validate: 2022-2023, test: 2024) prevent this leakage and simulate real-world deployment where you only have historical data.

### Why F2 instead of F1?

The cost of missing a delay (false negative) far exceeds the cost of a false alarm (false positive) in airline operations. F2 score weights recall twice as heavily as precision, aligning the optimization metric with the business objective.

### Why PageRank for airport importance?

Traditional approaches use airport size (number of flights) as a proxy for importance. PageRank captures something more subtle: how central an airport is in the flight network. A hub airport like ATL isn't just busy — it's a chokepoint where delays cascade to dozens of downstream flights. PageRank quantifies this network effect.

### Why hierarchical feature fallbacks?

New routes, new aircraft, or airports with limited history would have missing features. Instead of dropping these rows or using a single global average, we built a hierarchical system: aircraft-specific metrics → route-specific → airport-specific → global. This ensures every prediction has reasonable feature values while maximizing the use of available information.

---

## Lessons Learned

**Feature engineering is the real differentiator.** We spent more time engineering the 40+ features than training models. The rolling delay rates, weather Z-scores, and PageRank features each provided meaningful lift. No amount of model complexity compensates for weak features.

**Class imbalance is deceptive.** Our first logistic regression achieved 82% accuracy — which sounds good until you realize that predicting "no delay" for every flight gives you ~80% accuracy. Switching to F2 score and threshold tuning exposed the real challenge: catching the 20% of flights that are delayed.

**PySpark window functions are powerful but tricky.** Computing 30-day rolling delay rates across 90M rows required careful partitioning and ordering to avoid shuffles that would crash the cluster. Getting the window specifications right was a significant engineering challenge.

**Time-series splits reveal reality.** Our model performed noticeably worse on 2024 test data than on random hold-out sets. This is expected — real-world deployment faces distribution shift as flight patterns, weather patterns, and airline operations evolve. The time-series split gives a more honest estimate of production performance.

---

## Project Structure

```
Flight-Delay-Prediction/
├── Code/
│   └── w261_flight_prediction.ipynb    # Main analysis notebook (comprehensive)
├── Written_Report/
│   └── 261_Final_Report.ipynb          # Final written report
├── Presentation-Slides/
│   └── Phase 3 W261_Presentation.pdf   # Project presentation
├── assets/
│   └── W261_Phase_2_presentation.png   # Presentation visual
└── README.md                           # This file
```
