# ML-0027 — Human Development Index (HDI) Predictor

A full-stack machine learning project that predicts a country's Human Development
Index (HDI) score from four core indicators — life expectancy, mean years of
schooling, expected years of schooling, and GNI per capita — using a Linear
Regression model trained in a Jupyter notebook and served through a Flask web
application.

## Project Structure

```
ML - 0027 - Human Development Index/
├── Dataset/
│   ├── HDI.csv               # Country-level training dataset (167 countries)
│   └── generate_dataset.py   # Script used to synthesize the dataset
├── Flask/
│   ├── app.py                # Flask application (routes, API, validation)
│   ├── HDI.pkl                # Trained Linear Regression model (pickled)
│   ├── requirements.txt
│   ├── static/
│   │   └── style.css
│   └── templates/
│       ├── base.html
│       ├── index.html        # Home / overview page
│       ├── predict.html      # Prediction form + result
│       ├── about.html        # Methodology page
│       └── 404.html
└── Training/
    ├── HumDevIndex.ipynb     # Full training notebook (EDA → model → pickle)
    └── requirements.txt
```

## Dataset

`Dataset/HDI.csv` contains 167 countries with the following columns:

| Column | Description |
|---|---|
| `Country` | Country name |
| `Life_Expectancy` | Life expectancy at birth (years) |
| `Mean_Years_Schooling` | Mean years of schooling, adults 25+ |
| `Expected_Years_Schooling` | Expected years of schooling for a child entering school |
| `GNI_Per_Capita` | Gross National Income per capita (PPP $) |
| `HDI_Score` | Human Development Index score (0–1) — target variable |
| `HDI_Category` | Development tier: Low / Medium / High / Very High |

A small number of missing values were intentionally introduced into the numeric
columns to exercise the missing-value-handling step covered in the notebook.

> **Note:** This dataset is synthetically generated to realistically resemble
> published HDI-style statistics (UNDP Human Development Report ranges) for
> training and educational purposes. It is not an authoritative or official
> dataset — see the "Methodology" page in the app for details.

## 1. Environment Setup

### Training notebook

```bash
cd Training
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook HumDevIndex.ipynb
```

Run all cells top to bottom. The final cells save the trained model to
`../Flask/HDI.pkl`, overwriting the one already included in this repo if you
retrain.

### Flask app

```bash
cd Flask
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The app starts in debug mode at **http://127.0.0.1:5000**.

## 2. Notebook Workflow (`Training/HumDevIndex.ipynb`)

1. **Library imports** — NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, Pickle
2. **Dataset loading & understanding** — shape, dtypes, summary statistics, category counts
3. **Visualization** — distribution plot, strip plot by tier, correlation heatmap, scatter plots, pairplot
4. **Preprocessing** — missing value check + mean imputation, label encoding of `HDI_Category` (for reference), feature/target selection
5. **Train/test split** — 80/20 split, `random_state=42`
6. **Model training** — `LinearRegression` from scikit-learn
7. **Evaluation** — MAE, MSE, RMSE, R², actual-vs-predicted plot, residual distribution, feature coefficients
8. **Model saving** — serialized with `pickle` to `Flask/HDI.pkl`, with a reload sanity check

## 3. Flask Application

### Pages

| Route | Description |
|---|---|
| `GET /` | Home page — project overview, dimensions of HDI |
| `GET /predict` | Prediction form |
| `POST /predict` | Submits form, renders predicted score + tier |
| `GET /about` | Methodology / how the model works |

### REST API

**`POST /api/predict`**

Request body (JSON):
```json
{
  "life_expectancy": 78.5,
  "mean_schooling": 11.5,
  "expected_schooling": 15.8,
  "gni_per_capita": 42000
}
```

Response (200):
```json
{
  "hdi_score": 0.868,
  "hdi_category": "Very High",
  "inputs": {
    "life_expectancy": 78.5,
    "mean_schooling": 11.5,
    "expected_schooling": 15.8,
    "gni_per_capita": 42000.0
  }
}
```

Error response (400) — missing or invalid fields:
```json
{ "error": "Missing fields: gni_per_capita" }
```

Example with `curl`:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"life_expectancy": 55, "mean_schooling": 3.5, "expected_schooling": 8, "gni_per_capita": 1500}'
```

### Validation rules

| Field | Valid range |
|---|---|
| `life_expectancy` | 0 – 100 |
| `mean_schooling` | 0 – 20 |
| `expected_schooling` | 0 – 25 |
| `gni_per_capita` | 0 (exclusive) – 200,000 |

Predicted scores are clamped to the `[0, 1]` range before being classified into
a tier (Low < 0.550 ≤ Medium < 0.700 ≤ High < 0.800 ≤ Very High).

## 4. Retraining the Model

To retrain on updated data, replace `Dataset/HDI.csv` (keeping the same column
names) and re-run `Training/HumDevIndex.ipynb` end to end. The last cells will
overwrite `Flask/HDI.pkl` automatically — restart the Flask app afterward to
pick up the new model.

## Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (Linear Regression), Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML, CSS, Jinja2 templates (vanilla, no JS framework)
- **Model serialization:** Pickle
