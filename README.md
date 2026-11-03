# 🏙️ PropVista India - Real Estate Investment Advisor

Predicts whether a property is a **Good Investment** (classification) and its **price after 5 years** (regression),
with a purple-themed multi-page Streamlit dashboard.

## Files
| File | Purpose |
|---|---|
| `india_housing_prices.csv` | Raw dataset (250,000 rows, 23 columns) |
| `clean_data.py` | Cleaning + feature engineering → `cleaned_housing_data.csv` |
| `cleaned_housing_data.csv` | Clean data with engineered features and both targets |
| `eda.py` | Answers the 20 EDA questions → `eda_charts/` + `Insights.txt` |
| `Insights.txt` | Plain-text key insights |
| `app.py` | Streamlit app (7 pages, live filters, interactive Plotly charts, models) |
| `.streamlit/config.toml` | Dark-violet theme |
| `requirements.txt` | Dependencies |

## Run
```bash
pip install -r requirements.txt
python clean_data.py      # creates cleaned_housing_data.csv
python eda.py             # creates eda_charts/ and Insights.txt
streamlit run app.py
```
Set your name in `app.py`: `CREATOR_NAME = "Your Name"`.

## Methodology
1. **Cleaning** - trim text, drop duplicates, impute missing values, fix `Floor_No > Total_Floors` (46% of rows), recompute `Age_of_Property` and `Price_per_SqFt_INR`, flag IQR outliers (kept).
2. **Features** - Amenity_Count, Transport_Score, School_Density_Score, Infrastructure_Score (0-10), Age/Size categories.
3. **Good_Investment label** - 1 if at least 4 of 5 rules hold: price/sq ft ≤ city median, infrastructure ≥ median, transport Medium/High, Ready to move, age ≤ 15 years.
4. **Future_Price_5Y_Lakhs** - the dataset is a single snapshot, so growth is *assumed*: 5% base + up to 3% infrastructure + 1.5% transport + 1% new build + 1% under-construction, compounded 5 years.
5. **Models** - Random Forest classifier and regressor (one-hot + passthrough pipeline), trained on a 40,000-row sample inside the app; accuracy, F1, ROC-AUC, R², RMSE, MAE and feature importance are shown.

## Important notes
- The dataset has no crime-rate or true price history, so Infrastructure_Score replaces crime rate and future prices are scenario-based estimates.
- Because the label and future price are rule-derived, model scores are very high; retrain on real appreciation data for production use.
- `cleaned_housing_data.csv` is ~58 MB (under GitHub's 100 MB limit).
- MLflow (mentioned in the brief) is not wired in; add `mlflow.log_metric` calls inside `train_models()` if required.
