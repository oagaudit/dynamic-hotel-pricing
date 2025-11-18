# ============================================================
# run_feature_selection.py (COMPLETE FIXED VERSION)
# ============================================================
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.base import BaseEstimator, TransformerMixin
import re

# ------------------------------------------------------------
# Logging & Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)

log_file = LOG_DIR / "feature_selection_final.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

FEATURED_HOTEL_FILE = PROCESSED_DATA_DIR / "hotel_bookings_featured.csv"
PIPELINE_FILE = MODEL_DIR / "feature_selection_pipeline_final.pkl"
SELECTED_FEATURES_FILE = PROCESSED_DATA_DIR / "selected_features_list_final.csv"

TARGET = "adr"

# ------------------------------------------------------------
# Custom Transformer
# ------------------------------------------------------------
class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for col in X_df.columns:
            q1, q3 = X_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor*iqr, q3 + self.factor*iqr)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = np.clip(X_df[col], lower, upper)
        return X_df

# ------------------------------------------------------------
# Load Data and FILTER ALL DATE FEATURES BEFORE processing
# ------------------------------------------------------------
logger.info("Loading data...")
df = pd.read_csv(FEATURED_HOTEL_FILE)
y = df[TARGET]
X = df.drop(columns=[TARGET])

logger.info("FILTERING OUT ALL DATE/TIME-RELATED features BEFORE processing...")

# FIX 1: Delete all columns name have date, time 
date_time_keywords = ['date', 'time']

features_to_remove = []
for col in X.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in date_time_keywords):
        # Except meaningful business features 
        if not any(protected in col_lower for protected in [
            'lead_time', 'arrival_month_sin', 'arrival_month_cos', 
            'arrival_dow_sin', 'arrival_dow_cos'
        ]):
            features_to_remove.append(col)

logger.info(f"Removing {len(features_to_remove)} date/time-related features:")
for feat in features_to_remove:
    logger.info(f"  - {feat}")

# Use X_filtered replace process
X_filtered = X.drop(columns=features_to_remove)
logger.info(f"Original features: {X.shape[1]}, After filtering: {X_filtered.shape[1]}")

remaining_date_features = [col for col in X_filtered.columns if any(word in col.lower() for word in ['date', 'time'])]
if remaining_date_features:
    logger.warning(f"Still have date-like features: {remaining_date_features}")
else:
    logger.info("No date features remaining!")

# ------------------------------------------------------------
# Identify numeric & categorical features FROM FILTERED DATA
# ------------------------------------------------------------
numeric_features = X_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_filtered.select_dtypes(include=['object', 'category']).columns.tolist()

logger.info(f"Filtered - Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)}")

# ------------------------------------------------------------
# Manual feature priority for business (UPDATED)
# ------------------------------------------------------------
business_priority_features = [
    'competitor_avg_price', 'price_percentile', 'market_demand_index',
    'lead_time', 'total_nights', 'is_weekend_arrival', 'arrival_month_sin',
    'adr_lag_7d', 'adr_rolling_mean_7d', 'stays_in_week_nights',
    'hotel_premium_index', 'advance_booking', 'room_nights',
    # FIX 2: Change real features in data 
    'arrival_dow_sin', 'loyalty_score', 'cancellation_risk_score',
    'booking_velocity', 'adr_rolling_std_7d'  # add features 
]

available_priority_features = [f for f in business_priority_features if f in X_filtered.columns]
missing_priority_features = [f for f in business_priority_features if f not in X_filtered.columns]

logger.info(f"Available priority features: {len(available_priority_features)}/{len(business_priority_features)}")
if missing_priority_features:
    logger.warning(f"Missing priority features: {missing_priority_features}")

# แสดง features ที่มีอยู่จริง (10 features แรก)
logger.info("Available features in filtered data (first 15):")
for i, col in enumerate(X_filtered.columns[:15]):
    logger.info(f"  {i+1}. {col}")

# ------------------------------------------------------------
# Build Pipeline  
# ------------------------------------------------------------
numeric_transformer = Pipeline([
    ('outlier_clipper', OutlierClipper(factor=1.5)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ------------------------------------------------------------
# Full Pipeline with Lasso
# ------------------------------------------------------------
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selector', SelectFromModel(
        LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=5000),
        threshold='median'
    ))
])

# ------------------------------------------------------------
# Fit Pipeline WITH FILTERED DATA
# ------------------------------------------------------------
logger.info("Fitting pipeline with CLEAN filtered data...")
full_pipeline.fit(X_filtered, y)
logger.info("Pipeline fitting complete!")

# ------------------------------------------------------------
# Extract transformed feature names FROM FILTERED DATA
# ------------------------------------------------------------
ohe_feature_names = full_pipeline.named_steps['preprocessor'] \
    .named_transformers_['cat'] \
    .named_steps['onehot'] \
    .get_feature_names_out(categorical_features)
all_transformed_features = np.concatenate([numeric_features, ohe_feature_names])

logger.info(f"All transformed features: {len(all_transformed_features)}")

# ------------------------------------------------------------
# Get selected features after Lasso
# ------------------------------------------------------------
selector = full_pipeline.named_steps['feature_selector']
selected_mask = selector.get_support()
selected_features = all_transformed_features[selected_mask]

logger.info(f"Initially selected features: {len(selected_features)}")

# ------------------------------------------------------------
# Keep top 20 by absolute coefficient
# ------------------------------------------------------------
coefs = selector.estimator_.coef_
coef_values = np.abs(coefs)

top_n = 20
if len(coef_values) > top_n:
    top_indices = np.argsort(coef_values)[-top_n:]
    selected_features = all_transformed_features[top_indices]

logger.info(f"Selected top {len(selected_features)} features.")

# ------------------------------------------------------------
# VALIDATE: no date features
# ------------------------------------------------------------
logger.info("Final validation of selected features...")

date_pattern = re.compile(r'.*date.*', re.IGNORECASE)
date_features_in_selected = [f for f in selected_features if date_pattern.search(str(f))]
meaningful_features = [f for f in selected_features if not date_pattern.search(str(f))]

if date_features_in_selected:
    logger.error(f"STILL HAVE DATE FEATURES: {date_features_in_selected}")
    selected_features = meaningful_features
    logger.info(f"Removed date features, now have {len(selected_features)} meaningful features")
else:
    logger.info(f"Clean! Meaningful features selected: {len(meaningful_features)}")

# Show top features
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': coef_values[top_indices] if len(coef_values) > top_n else coef_values[selected_mask]
}).sort_values('importance', ascending=False)

logger.info("Top 10 most important features:")
for idx, row in feature_importance.head(10).iterrows():
    logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

# ------------------------------------------------------------
# Export CLEAN features
# ------------------------------------------------------------
pd.DataFrame({"feature": selected_features}).to_csv(SELECTED_FEATURES_FILE, index=False)
joblib.dump(full_pipeline, PIPELINE_FILE)

logger.info(f"Saved pipeline to {PIPELINE_FILE}")
logger.info(f"Saved {len(selected_features)} CLEAN features to {SELECTED_FEATURES_FILE}")

# FINAL COMPARISON
logger.info("\n" + "="*60)
logger.info("FINAL FEATURE SELECTION RESULTS")
logger.info("="*60)
logger.info(f"Original dataset features: {X.shape[1]}")
logger.info(f"After removing date features: {X_filtered.shape[1]}")
logger.info(f"Final selected features: {len(selected_features)}")
logger.info(f"All features are meaningful: {len(meaningful_features) == len(selected_features)}")

# Save feature importance
importance_file = PROCESSED_DATA_DIR / "feature_importance_final.csv"
feature_importance.to_csv(importance_file, index=False)
logger.info(f"Feature importance saved to: {importance_file}")

# FINAL VALIDATION
logger.info("\n" + "="*60)
logger.info("FINAL VALIDATION")
logger.info("="*60)
logger.info("Selected features:")
for i, feat in enumerate(selected_features, 1):
    logger.info(f"  {i}. {feat}")

# Check if we have key business features
key_business_features = ['competitor_avg_price', 'market_demand_index', 'price_percentile']
missing_business_features = [f for f in key_business_features if f not in selected_features]

if not missing_business_features:
    logger.info("SUCCESS: All key business features are included!")
else:
    logger.warning(f"Missing key business features: {missing_business_features}")

if not any(date_pattern.search(str(f)) for f in selected_features):
    logger.info("SUCCESS: No date-specific features in final selection!")
else:
    logger.error("FAILED: Still contains date features!")