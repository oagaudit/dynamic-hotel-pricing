import os
import pandas as pd
import kagglehub
from config.paths import RAW_HOTEL_FILE, CLEANED_HOTEL_FILE 

def download_hotel_booking_dataset():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")

    file_src = os.path.join(path, "hotel_bookings.csv")
    if not os.path.exists(RAW_HOTEL_FILE):
        pd.read_csv(file_src).to_csv(RAW_HOTEL_FILE, index=False)
        print(f"Dataset saved to: {RAW_HOTEL_FILE}")
    else:
        print(f"Using existing dataset at: {RAW_HOTEL_FILE}")

    return pd.read_csv(RAW_HOTEL_FILE)


def clean_hotel_booking_data(df):
    """Perform data cleaning and preprocessing on the hotel booking dataset."""
    pd.set_option('display.max_columns', None)

    # Drop irrelevant column
    df = df.drop(['company'], axis=1)

    # Fill missing values
    df.fillna({
        'agent': df['agent'].median(),
        'country': df['country'].mode()[0],
        'children': df['children'].mode()[0]
    }, inplace=True)

    # Convert data types
    df['children'] = df['children'].fillna(0).astype(int)
    df['agent'] = df['agent'].astype('Int64')
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')

    # Handle outliers in ADR (Average Daily Rate)
    Q1 = df['adr'].quantile(0.25)
    Q3 = df['adr'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df['adr'] > 0) & (df['adr'] < 5000)]
    df = df[(df['adr'] >= lower_bound) & (df['adr'] <= upper_bound)]

    # Remove bookings with no guests
    df = df[df['adults'] >= 1]

    # Limit total stay duration
    df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
    df = df[df['total_nights'] <= 30]

    # Cap waiting list days
    df['days_in_waiting_list'] = df['days_in_waiting_list'].clip(upper=180)

    # Save cleaned dataset to CSV
    df.to_csv(CLEANED_HOTEL_FILE, index=False)
    print(f"Cleaned dataset saved to: {CLEANED_HOTEL_FILE}")
    return df


if __name__ == "__main__":
    df = download_hotel_booking_dataset()
    df = clean_hotel_booking_data(df)

    print("\nCleaned dataset overview:")
    print(df.info())
    print(df.describe())
