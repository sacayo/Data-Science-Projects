# Google Customer Return Prediction

> Predicting returning customers for Google's online merchandise store using LSTM neural networks on 35GB of sequential user behavior data — achieving 93.72% recall, a 20.65 percentage point improvement over baseline.

## Collaborators

[Jasmine Lau](https://github.com/AppleTater)
[Diego Moss](https://github.com/Mossd-2)
[Roz Huang](https://github.com/ronghuang0604)
[Connor Huh](https://github.com/ConorHuh)

---

## 01. The Problem

### Context

Google's online merchandise store generates millions of user sessions per month, but only a fraction of visitors make repeat purchases. For every dollar spent on customer acquisition, the return on investment depends heavily on whether that customer comes back. If you can identify which users are likely to return before they do, you can target marketing spend on users who need a nudge rather than wasting it on users who would return anyway — or users who will never return regardless.

### Challenge

We needed to build a model that:
- **Processes sequential behavior**: User activity unfolds over time — a user who visits 5 times in a week behaves differently from one who visits once a month. The model must understand temporal patterns, not just aggregate statistics.
- **Handles 35GB of raw data**: Google Analytics data stored in BigQuery as deeply nested JSON, requiring significant ETL before modeling.
- **Prevents temporal data leakage**: When predicting whether a user returns in month 8, the model must only see data from months 1-7. Using future data to predict the past gives artificially inflated metrics that collapse in production.
- **Achieves high recall**: A missed returning customer (false negative) represents a lost marketing opportunity. The cost of incorrectly flagging a non-returning user as returning (false positive) is just a wasted marketing email — much cheaper than missing a high-value customer entirely.
- **Runs fast at inference**: A model that takes minutes per prediction is useless for real-time marketing decisions. We needed sub-second inference per user.

The fundamental modeling question: can you predict future behavior from a sequence of past sessions, where users have variable-length histories and different engagement patterns?

---

## 02. The Approach

### LSTM Model Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Dual-Input LSTM Architecture                   │
│                                                                  │
│  Dynamic Features                     Stable Features            │
│  (Time-Series: 13 months)             (User Averages)            │
│                                                                  │
│  ┌────────────────────┐               ┌──────────────────┐      │
│  │  Month 1 features  │               │ Avg visits       │      │
│  │  Month 2 features  │               │ Avg page views   │      │
│  │  Month 3 features  │               │ Avg session time │      │
│  │  ...               │               │ Avg transactions │      │
│  │  Month 13 features │               │ Device type      │      │
│  └─────────┬──────────┘               │ Traffic source   │      │
│            │                          │ Browser          │      │
│            ▼                          │ Location         │      │
│  ┌────────────────────┐               └────────┬─────────┘      │
│  │   Masking Layer    │                        │                 │
│  │  (variable-length  │                        │                 │
│  │   sequences)       │                        │                 │
│  └─────────┬──────────┘                        │                 │
│            │                                   │                 │
│            ▼                                   │                 │
│  ┌────────────────────┐                        │                 │
│  │   LSTM Layer       │                        │                 │
│  │   (50 units)       │                        │                 │
│  └─────────┬──────────┘                        │                 │
│            │                                   │                 │
│            ▼                                   │                 │
│  ┌────────────────────┐                        │                 │
│  │   Dropout (0.2)    │                        │                 │
│  └─────────┬──────────┘                        │                 │
│            │                                   │                 │
│            └───────────────┬───────────────────┘                 │
│                            │                                     │
│                            ▼                                     │
│                 ┌────────────────────┐                            │
│                 │   Concatenate      │                            │
│                 │  (LSTM output +    │                            │
│                 │   stable features) │                            │
│                 └─────────┬──────────┘                            │
│                           │                                      │
│                           ▼                                      │
│                 ┌────────────────────┐                            │
│                 │   Dense (50)       │                            │
│                 │   ReLU activation  │                            │
│                 └─────────┬──────────┘                            │
│                           │                                      │
│                           ▼                                      │
│                 ┌────────────────────┐                            │
│                 │   Dense (1)        │                            │
│                 │   Sigmoid          │                            │
│                 │   → P(return)      │                            │
│                 └────────────────────┘                            │
│                                                                  │
│  Total Parameters: 50,255                                        │
│  Inference Time: ~2ms per user                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Data Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Google BigQuery │     │  JSON Flattening  │     │  Feature         │
│  (35GB raw data) │────▶│  & Aggregation    │────▶│  Engineering     │
│                  │     │                   │     │  (27 features)   │
│  Aug 2016 -      │     │  Nested GA data   │     │                  │
│  Aug 2017        │     │  → flat tables    │     │  Numeric:        │
│  (daily tables)  │     │  → monthly agg    │     │  visits, views,  │
│                  │     │                   │     │  session quality │
└──────────────────┘     └──────────────────┘     │                  │
                                                   │  Categorical:    │
                                                   │  device, source, │
                                                   │  browser, OS,    │
                                                   │  location        │
                                                   └────────┬─────────┘
                                                            │
                                                            ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Stratified      │     │  Class Balance   │◀────│  Sequence        │
│  Train/Val/Test  │◀────│  (equal pos/neg) │     │  Creation        │
│  60% / 20% / 20% │     │                  │     │                  │
│                  │     │  Keep all returns │     │  [T1]            │
└──────────────────┘     │  + equal non-ret  │     │  [T1, T2]        │
                         └──────────────────┘     │  [T1, T2, T3]    │
                                                   │  ...              │
                                                   │  [T1...T13]       │
                                                   │                  │
                                                   │  Prevents future │
                                                   │  data leakage    │
                                                   └──────────────────┘
```

### Sequential Modeling: Why LSTM?

The key insight driving this project: **user behavior is sequential, not static.** A traditional model (logistic regression, random forest) treats each user as a bag of features — total visits, average page views, etc. But the pattern of behavior over time tells a richer story:

- A user whose visits are **increasing** month over month is likely engaged and will return
- A user whose visits **spiked once then dropped** may have been a one-time buyer
- A user who **consistently visits but never purchases** has different return probability than a user who **purchases immediately**

LSTMs (Long Short-Term Memory networks) capture these temporal patterns through their gated architecture — they learn which past signals to remember and which to forget.

### Preventing Temporal Data Leakage

This is the most critical aspect of the data preparation. We created time-aware sequences that prevent future information from leaking into predictions:

```
User A's data across 13 months:

Prediction for Month 2:  [Month 1] → will they return in Month 2+?
Prediction for Month 5:  [Month 1, 2, 3, 4] → will they return in Month 5+?
Prediction for Month 13: [Month 1, 2, ... 12] → will they return in Month 13+?
```

Each prediction only uses data from **before** the prediction window. A model that accidentally uses Month 5 data to predict Month 3 behavior would achieve artificially high metrics that collapse in production.

### Feature Engineering (27 Features)

**Numeric Features**:
- Visits, page views, screen views per month
- Session quality score, time on site
- Transaction revenue, transaction count
- E-commerce actions: product detail views, add-to-cart, remove-from-cart, checkout, purchases, refunds

**Categorical Features** (encoded):
- Traffic source (organic, paid, direct, referral, social)
- Device type (desktop, mobile, tablet)
- Browser, operating system
- Location (continent, country, region, city)

### Handling Class Imbalance

Only ~15-20% of users return in any given month. Training on the raw distribution would produce a model that predicts "won't return" for everyone and achieves 80%+ accuracy while being completely useless.

Our approach: keep all positive cases (returning users) and sample an equal number of negative cases (non-returning users). This creates a balanced training set where the model must learn to distinguish between the two classes rather than defaulting to the majority class.

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Source** | Google BigQuery | 35GB of Google Analytics data |
| **Data Processing** | BigQuery SQL, Pandas | ETL, JSON flattening, aggregation |
| **Deep Learning** | TensorFlow/Keras | LSTM model architecture and training |
| **Baseline Model** | Scikit-learn | Logistic regression for comparison |
| **Sequence Handling** | NumPy | Time-aware sequence creation with masking |
| **Hyperparameter Tuning** | Grid search | Systematic exploration of LSTM configurations |
| **Visualization** | Matplotlib, Seaborn | ROC curves, confusion matrices, learning curves |

---

## 03. Results

### Model Comparison

| Metric | Baseline (Logistic Regression) | LSTM |
|--------|-------------------------------|------|
| Accuracy | 79.74% | **86.01%** |
| Precision | **84.31%** | 81.20% |
| **Recall** | 73.07% | **93.72%** |
| AUC-ROC | ~0.90 | **~0.94** |

### Key Takeaway

The LSTM improved recall by **20.65 percentage points** over the logistic regression baseline (73.07% → 93.72%). This means the LSTM catches 93.72% of all returning customers, compared to only 73.07% for the baseline. In a customer base of 100,000 users where 20,000 will return, this difference means correctly identifying an additional ~4,130 returning customers.

The trade-off: precision dropped slightly (84.31% → 81.20%), meaning the LSTM flags more false positives. But in marketing, the cost of sending an unnecessary email to a non-returning customer is negligible compared to the cost of missing a returning customer entirely.

### Model Specifications

| Parameter | Value |
|-----------|-------|
| LSTM units | 50 |
| Dropout rate | 0.2 |
| Dense layer | 50 neurons (ReLU) |
| Optimizer | Adam (lr=0.001) |
| Batch size | 32 |
| Epochs | 10 |
| Total parameters | 50,255 |
| **Inference time** | **~2ms per user** |

### Hyperparameters Explored

| Parameter | Values Tested | Best |
|-----------|---------------|------|
| LSTM units | 50 | 50 |
| Dropout | 0.2, 0.5 | **0.2** |
| Dense layer | None, 50 | **50** |
| Epochs | 10, 15 | **10** |
| Learning rate | 0.001, 0.0001 | **0.001** |
| Batch size | 32 | 32 |

---

## Technical Decisions & Trade-offs

### Why LSTM instead of a simpler model (Random Forest, XGBoost)?

Traditional ML models treat features as a flat vector — they can't naturally model the sequential nature of user behavior over 13 months. An LSTM processes the sequence of monthly features in order, learning temporal patterns like increasing engagement, seasonal purchasing behavior, and declining interest. The 20.65pp recall improvement over logistic regression validates that temporal patterns carry significant predictive signal.

### Why a masking layer?

Users have variable-length histories. A user who started visiting in Month 5 only has 9 months of data, not 13. Without masking, the LSTM would process zero-padded months as if they were real data (zero visits, zero page views), potentially learning misleading patterns. The masking layer tells the LSTM to ignore padded timesteps entirely.

### Why recall as the primary metric?

False negatives (missing a returning customer) cost more than false positives (flagging a non-returner). A missed returning customer is a lost marketing opportunity — they might not receive the targeted promotion, loyalty reward, or re-engagement email that would have increased their lifetime value. A false positive just means sending an extra marketing email to someone who won't act on it — the marginal cost is near zero.

### Why dual-input architecture (dynamic + stable features)?

Some user characteristics are inherently temporal (visits per month, transaction amounts) while others are relatively stable (device type, traffic source, geographic location). Feeding stable features through the LSTM would force the network to learn that they don't change — wasting capacity. The dual-input design lets the LSTM focus on temporal patterns while the stable features provide user-level context at the decision layer.

### Why class balancing instead of class weighting?

We tried both approaches. Balancing (equal positive/negative samples) produced more stable training and better-calibrated probabilities than class weighting (keeping all data but weighting the loss function). This is likely because the class weighting approach still exposes the model to a flood of negative examples per batch, making gradient updates noisy.

---

## Lessons Learned

**Sequential modeling captures what aggregation destroys.** A user with 50 total visits over 13 months could be a steady monthly visitor (likely to return) or a one-month power user who never came back. Aggregate features lose this distinction. The LSTM's ability to model visit patterns over time — not just totals — was the primary driver of its superior recall.

**Data leakage is subtle and devastating.** Our first attempt accidentally included future month data in the feature vectors, achieving 98% recall. We didn't catch it until we examined predictions for users who hadn't visited yet — the model was perfectly predicting their return because it could see their return data in the features. Time-aware sequence construction was essential.

**Simple models are strong baselines.** Logistic regression achieved 73% recall and 0.90 AUC with minimal tuning. The LSTM's improvement (93.72% recall, 0.94 AUC) was meaningful but not trivial to achieve. In many production settings, the logistic regression baseline would be "good enough" — the LSTM's advantage justifies the additional complexity only when the value of catching those extra 20% of returning customers is high.

**2ms inference time matters for real-time marketing.** With 50,255 parameters, the model is small enough to serve predictions in real-time. This enables use cases like triggering a pop-up promotion the moment a high-probability returning customer loads the homepage, rather than waiting for a batch prediction pipeline to run overnight.

---

## Project Structure

```
Google-Customer-Return-Prediction/
├── Code/
│   └── project_notebook.ipynb      # Main analysis notebook
├── Slides/
│   └── 207 Project Presentation.pdf  # Project presentation
├── assets/
│   └── 207 Project Presentation.png  # Presentation visual
├── LICENSE
└── README.md                        # This file
```
