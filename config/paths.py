import os
import pandas as pd  

# Project base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SELECTED_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, "selected_features_list_top50.csv") 

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths
RAW_HOTEL_FILE = os.path.join(RAW_DATA_DIR, "hotel_bookings.csv")
CLEANED_HOTEL_FILE = os.path.join(PROCESSED_DATA_DIR, "hotel_bookings_cleaned.csv")
FEATURED_HOTEL_FILE = os.path.join(PROCESSED_DATA_DIR, "hotel_bookings_featured.csv")
PIPELINE_FILE = os.path.join(MODEL_DIR, "feature_selection_pipeline.pkl")


# -------------------------------
# Data loader function
# -------------------------------
def load_hotel_dataset(cleaned: bool = True) -> pd.DataFrame:
    """
    Load hotel booking dataset.
    - cleaned=True: Load cleaned CSV if exists, else fallback to raw + warning.
    - cleaned=False: Load raw CSV, download from Kaggle if missing.
    """
    if cleaned:
        if os.path.exists(CLEANED_HOTEL_FILE):
            print(f"Loading cleaned dataset from: {CLEANED_HOTEL_FILE}")
            return pd.read_csv(CLEANED_HOTEL_FILE)
        else:
            print(f"Warning: Cleaned file not found, loading raw dataset instead.")
    
    # Load raw dataset
    if not os.path.exists(RAW_HOTEL_FILE):
        print("Downloading raw dataset from Kaggle...")
        path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")
        file_src = os.path.join(path, "hotel_bookings.csv")
        pd.read_csv(file_src).to_csv(RAW_HOTEL_FILE, index=False)
        print(f"Raw dataset saved to: {RAW_HOTEL_FILE}")
    
    print(f"Loading raw dataset from: {RAW_HOTEL_FILE}")
    return pd.read_csv(RAW_HOTEL_FILE)