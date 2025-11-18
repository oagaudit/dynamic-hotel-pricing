import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.paths import MODEL_DIR, LOG_DIR, PROCESSED_DATA_DIR, FEATURED_HOTEL_FILE
from src.run_feature_selection import OutlierClipper  # custom transformer

# ============================================================
# Logging Setup
# ============================================================
LOG_DIR = Path(LOG_DIR)
LOG_DIR.mkdir(exist_ok=True, parents=True)
log_file = LOG_DIR / "xgboost_baseline_final.log"  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration (UPDATED FILE NAMES)
# ============================================================
TARGET = "adr"
PIPELINE_FILE = Path(MODEL_DIR) / "feature_selection_pipeline_final.pkl" 
SELECTED_FEATURES_FILE = Path(PROCESSED_DATA_DIR) / "selected_features_list_final.csv"  
BASELINE_MODEL_FILE = Path(MODEL_DIR) / "xgboost_baseline_model_final.pkl"  

# ============================================================
# Load data & pipeline (FIXED VERSION)
# ============================================================
def load_data_and_pipeline():
    logger.info(f"Loading featured dataset from: {FEATURED_HOTEL_FILE}")
    df = pd.read_csv(FEATURED_HOTEL_FILE)
    logger.info(f"Dataset shape: {df.shape}")

    logger.info("Applying same date filtering as feature selection...")
    X = df.drop(columns=[TARGET])
    
    date_time_keywords = ['date', 'time']
    features_to_remove = []
    for col in X.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in date_time_keywords):
            if not any(protected in col_lower for protected in [
                'lead_time', 'arrival_month_sin', 'arrival_month_cos', 
                'arrival_dow_sin', 'arrival_dow_cos'
            ]):
                features_to_remove.append(col)
    
    X_filtered = X.drop(columns=features_to_remove)
    y = df[TARGET]
    
    logger.info(f"Using filtered data: {X_filtered.shape[1]} features (no date features)")

    if not PIPELINE_FILE.exists():
        logger.error(f"Pipeline file not found: {PIPELINE_FILE}")
        raise FileNotFoundError(f"Pipeline file not found: {PIPELINE_FILE}")
    
    logger.info(f"Loading pipeline from: {PIPELINE_FILE}")
    pipeline = joblib.load(PIPELINE_FILE)

    # load selected features
    if not SELECTED_FEATURES_FILE.exists():
        logger.error(f"Features file not found: {SELECTED_FEATURES_FILE}")
        raise FileNotFoundError(f"Features file not found: {SELECTED_FEATURES_FILE}")
    
    logger.info(f"Loading selected features from: {SELECTED_FEATURES_FILE}")
    selected_features_df = pd.read_csv(SELECTED_FEATURES_FILE)
    selected_features = selected_features_df["feature"].tolist()
    
    logger.info(f"Loaded {len(selected_features)} features for training")

    return X_filtered, y, pipeline, selected_features

# ============================================================
# Preprocess data (FIXED VERSION)
# ============================================================
def preprocess_data(X, y, pipeline, selected_features):
    logger.info("Applying preprocessing pipeline (transform)...")
    
    X_transformed = pipeline.named_steps["preprocessor"].transform(X)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    ohe_feature_names = pipeline.named_steps['preprocessor'] \
        .named_transformers_['cat'] \
        .named_steps['onehot'] \
        .get_feature_names_out(categorical_features)

    all_feature_names = np.concatenate([numeric_features, ohe_feature_names])
    X_df = pd.DataFrame(X_transformed, columns=all_feature_names)

    selected_features_mapped = [f for f in all_feature_names if f in selected_features]
    if len(selected_features_mapped) == 0:
        logger.warning("Selected features not found in transformed data. Using all features instead.")
        selected_features_mapped = all_feature_names

    logger.info(f"Number of features used for training: {len(selected_features_mapped)}")
    
    # แสดงตัวอย่าง features ที่ใช้
    logger.info("Sample features used for training:")
    for i, feat in enumerate(selected_features_mapped[:5]):
        logger.info(f"  {i+1}. {feat}")
    
    return X_df, y, selected_features_mapped

# ============================================================
# Train XGBoost
# ============================================================
def train_xgboost(X, y, selected_features_mapped):
    logger.info("Filtering selected features for XGBoost...")
    X_sel = X[selected_features_mapped]
    logger.info(f"Training with {X_sel.shape[1]} features...")

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.2, random_state=42
    )

    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    )

    logger.info("Training XGBoost model...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = xgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"XGBoost Results -> MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R²: {r2:.4f}")
    return xgb_model, mae, rmse, r2

# ============================================================
# Main (FIXED VERSION)
# ============================================================
def main():
    try:
        X, y, pipeline, selected_features = load_data_and_pipeline()
        X_processed, y, selected_features_mapped = preprocess_data(X, y, pipeline, selected_features)
        model, mae, rmse, r2 = train_xgboost(X_processed, y, selected_features_mapped)

        Path(MODEL_DIR).mkdir(exist_ok=True, parents=True)
        
        joblib.dump(model, BASELINE_MODEL_FILE)
        logger.info(f"Saved XGBoost model to: {BASELINE_MODEL_FILE}")

        print("\n" + "="*60)
        print("XGBOOST MODEL SUMMARY (CLEAN FEATURES) ")
        print("="*60)
        print(f" • Data shape: {X.shape[0]} rows, {X.shape[1]} features (filtered)")
        print(f" • Selected features used: {len(selected_features_mapped)}")
        print(f" • MAE:  ${mae:.2f}")
        print(f" • RMSE: ${rmse:.2f}")
        print(f" • R²:   {r2:.4f}")
        
        print(f"\n • Key Features: {selected_features[:3]}...")
        print(f" • Model saved to: {BASELINE_MODEL_FILE}")
        print("="*60)
        
    except Exception as e:
        logger.error(f" Training failed: {e}")
        print(f"\n Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()