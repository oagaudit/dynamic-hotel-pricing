"""
Feature Engineering Pipeline for Hotel Pricing Project
Creates 9 categories of engineered features for ADR prediction
"""

import pandas as pd
import numpy as np
import os
import sys
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dataset paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from config.paths import PROCESSED_DATA_DIR, CLEANED_HOTEL_FILE, FEATURED_HOTEL_FILE, SELECTED_FEATURES_FILE, PIPELINE_FILE, MODEL_DIR, LOG_DIR


class FeaturePipeline:

    def __init__(self):
        self.df = None
        self.feature_groups = {}

    def load_data(self):
        logger.info("Loading cleaned dataset...")
        self.df = pd.read_csv(CLEANED_HOTEL_FILE)
        logger.info(f"Loaded dataset: {self.df.shape}")
        return self.df

    # 1. Temporal & Seasonal Features
    def create_temporal_features(self, df):
        logger.info("Creating temporal features...")

        df['arrival_date'] = pd.to_datetime(df['reservation_status_date'])

        month_map = {
            'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
            'July':7,'August':8,'September':9,'October':10,'November':11,'December':12
        }
        df['arrival_month_num'] = df['arrival_date_month'].map(month_map)
        df['arrival_week_of_year'] = df['arrival_date'].dt.isocalendar().week
        df['arrival_day_of_week'] = df['arrival_date'].dt.dayofweek
        df['is_weekend_arrival'] = df['arrival_day_of_week'].isin([5,6]).astype(int)

        # Cyclical encoding
        df['arrival_month_sin'] = np.sin(2 * np.pi * df['arrival_month_num'] / 12)
        df['arrival_month_cos'] = np.cos(2 * np.pi * df['arrival_month_num'] / 12)

        df['arrival_dow_sin'] = np.sin(2 * np.pi * df['arrival_day_of_week'] / 7)
        df['arrival_dow_cos'] = np.cos(2 * np.pi * df['arrival_day_of_week'] / 7)

        # Peak season
        df['is_peak_season'] = df['arrival_month_num'].isin([6,7,8,12,1]).astype(int)

        self.feature_groups['temporal'] = [
            'arrival_month_num','arrival_week_of_year','arrival_day_of_week',
            'is_weekend_arrival','arrival_month_sin','arrival_month_cos',
            'arrival_dow_sin','arrival_dow_cos','is_peak_season'
        ]

        return df

    # 2. Lead Time Features
    def create_lead_time_features(self, df):
        logger.info("Creating lead time features...")
        df['lead_time_category'] = pd.cut(
            df['lead_time'],
            bins=[-1, 7, 30, 90, 500],
            labels=['last_minute','short','medium','long']
        )

        df['last_minute_booking'] = (df['lead_time'] < 7).astype(int)
        df['advance_booking'] = (df['lead_time'] > 90).astype(int)

        self.feature_groups['lead_time'] = [
            'lead_time_category','last_minute_booking','advance_booking'
        ]
        return df

    # 3. Demand & Occupancy
    def create_demand_features(self, df):
        logger.info("Creating demand features...")

        df['total_guests'] = df['adults'] + df['children'] + df['babies']
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        df['room_nights'] = df['total_guests'] * df['total_nights']

        df = df.sort_values('arrival_date')
        df['market_demand_index'] = df['total_guests'].rolling(7, min_periods=1).mean()
        df['booking_velocity'] = df['total_guests'].rolling(7, min_periods=1).sum() / 7

        self.feature_groups['demand'] = [
            'total_guests', 'total_nights', 'room_nights',
            'market_demand_index','booking_velocity'
        ]
        return df

    # 4. Customer Segmentation
    def create_customer_features(self, df):
        logger.info("Creating customer segmentation features...")

        df['is_family'] = ((df['children']>0) | (df['babies']>0)).astype(int)
        df['is_group'] = (df['adults'] > 2).astype(int)
        df['is_business'] = (df['customer_type'] == 'Contract').astype(int)
        df['loyalty_score'] = df['is_repeated_guest'] + df['previous_bookings_not_canceled']

        self.feature_groups['customer'] = [
            'is_family','is_group','is_business','loyalty_score'
        ]
        return df

    # 5. Room & Hotel Attributes
    def create_room_hotel_features(self, df):
        logger.info("Creating room & hotel features...")

        df['is_room_upgraded'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)
        df['hotel_type_encoded'] = df['hotel'].map({'City Hotel':0,'Resort Hotel':1})
        df['hotel_premium_index'] = df['hotel_type_encoded'] + df['is_peak_season']

        self.feature_groups['room_hotel'] = [
            'is_room_upgraded','hotel_type_encoded','hotel_premium_index'
        ]
        return df

    # 6. Price Competition
    def create_competitive_features(self, df):
        logger.info("Creating competitive features...")
        np.random.seed(42)
        num_comp = 5

        competitor_prices = [
            df['adr'] * (1 + np.random.uniform(-0.15, 0.15, len(df)))
            for _ in range(num_comp)
        ]

        comp_df = pd.concat(competitor_prices, axis=1)
        comp_df.columns = [f'competitor_{i+1}_price' for i in range(num_comp)]
        df = pd.concat([df, comp_df], axis=1)

        df['competitor_avg_price'] = comp_df.mean(axis=1)
        df['price_percentile'] = df['adr'].rank(pct=True)

        self.feature_groups['competitive'] = comp_df.columns.tolist() + [
            'competitor_avg_price','price_percentile'
        ]
        return df

    # 7. Cancellation & Risk
    def create_risk_features(self, df):
        logger.info("Creating risk features...")

        df['deposit_type_numeric'] = df['deposit_type'].map({'No Deposit':0,'Non Refund':0,'Refundable':1})
        df['cancellation_risk_score'] = df['previous_cancellations'] + df['booking_changes']

        self.feature_groups['risk'] = ['deposit_type_numeric','cancellation_risk_score']
        return df

    # 8. Interaction Features
    def create_interaction_features(self, df):
        logger.info("Creating interaction features...")

        df['lead_time_x_total_nights'] = df['lead_time'] * df['total_nights']
        df['is_weekend_x_is_family'] = df['is_weekend_arrival'] * df['is_family']
        df['deposit_x_cancellation_risk'] = df['deposit_type_numeric'] * df['cancellation_risk_score']

        self.feature_groups['interaction'] = [
            'lead_time_x_total_nights','is_weekend_x_is_family','deposit_x_cancellation_risk'
        ]
        return df

    # 9. Time-series Lag Features
    def create_time_series_features(self, df):
        logger.info("Creating time-series features...")

        df = df.sort_values('arrival_date')
        df['adr_lag_7d'] = df['adr'].shift(7).bfill()
        df['adr_rolling_mean_7d'] = df['adr'].rolling(7,min_periods=1).mean()

        self.feature_groups['time_series'] = ['adr_lag_7d','adr_rolling_mean_7d']
        return df

    # Run full pipeline
    def run_pipeline(self):
        self.load_data()

        feature_steps = [
            self.create_temporal_features,
            self.create_lead_time_features,
            self.create_demand_features,
            self.create_customer_features,
            self.create_room_hotel_features,
            self.create_competitive_features,
            self.create_risk_features,
            self.create_interaction_features,
            self.create_time_series_features
        ]

        for step in feature_steps:
            self.df = step(self.df)
            logger.info(f"Applied: {step.__name__}")

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        self.df.to_csv(FEATURED_HOTEL_FILE, index=False)

        logger.info(f"\n Feature engineering complete!")
        logger.info(f" Final dataset shape: {self.df.shape}")
        return self.df


def run_feature_pipeline():
    pipeline = FeaturePipeline()
    return pipeline.run_pipeline()


if __name__ == "__main__":
    run_feature_pipeline()
    print(" Saved featured dataset successfully!")
