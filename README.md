## Introduction

This project is the Capstone for my Master's degree in Data Science.

The **Dynamic Hotel Pricing Model (DPM)** addresses the limitations of traditional, fixed pricing strategies in hotel management. We developed an intelligent system that uses **Machine Learning (XGBoost)** to predict the optimal Average Daily Rate (ADR). Furthermore, it integrates **Explainable AI (XAI)** techniques and a **Large Language Model (LLM)** to provide transparent, actionable pricing recommendations based on real-time market dynamics, competitor behavior, and seasonal trends. All insights are visualized and presented through an interactive Streamlit dashboard.

---

## Model Performance & Evaluation

The model performance is evaluated using standard regression metrics, showing **exceptional performance** on the held-out test set:

* **Mean Absolute Error (MAE):** $0.22 (The average error is only $0.22)
* **Root Mean Square Error (RMSE):** $0.46
* **Coefficient of Determination (R²):** $0.9999

---

## Top 20 Features Driving ADR

The feature selection process identified 20 core predictors. These features fall into key categories, enabling the model to capture market complexities:

* **Competitive & Price Dynamics:** `competitor_avg_price`, `price_percentile`, `adr_rolling_mean_7d`, `adr_lag_7d`, `hotel_premium_index`.
* **Booking Behavior & Stay:** `room_nights`, `stays_in_week_nights`, `lead_time_x_total_nights`, `advance_booking`, `is_canceled`.
* **Segmentation & Origin:** `market_segment_Groups`, `market_segment_Offline TA/TO`, and specific country indicators (`country_CZE`, `DEU`, `IRQ`, `DMA`).
* **Hotel & Temporal:** `reserved_room_type_A`, `hotel_type_encoded`, `arrival_month_sin` (for seasonality).

### Top 5 Features by SHAP Technique

SHAP analysis provides local explainability for each price prediction. For a sample prediction, the 5 most influential features were:

1. `price_percentile`
2. `stays_in_week_nights`
3. `room_nights`
4. `market_segment_Groups`
5. `advance_booking`

---

## Installation & Setup

### **1. Clone Repository & Setup Environment**

```bash
git clone https://github.com/oagaudit/dynamic-hotel-pricing.git
cd dynamic-hotel-pricing

# Create and activate virtual environment
python3 -m venv hotel_venv
source hotel_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### **2. Setup Ollama LLM (for Strategic Recommendations)**

You must have the Ollama server installed and running locally.
Download Ollama from this link first : https://ollama.com/library/llama3.2:1b
```bash
ollama pull llama3.2:1b
ollama list
```

> Ensure the Ollama server is running (default: `http://localhost:11434`)  

---

## 3. Pipeline Execution

Run the following scripts sequentially to process data, train the model, and prepare the pricing engine.

| #  | Script                                  | Description                                                    | Output/Result                                |
| -- | --------------------------------------- | -------------------------------------------------------------- | -------------------------------------------- |
| 1  | `python -m src.data_collect_cleansing`  | Downloads data from Kaggle and performs initial data cleaning. | `hotel_bookings_cleaned.csv`                 |
| 2  | `python -m src.feature_engineering`     | Creates over 70 complex features for predictive power.         | `hotel_bookings_featured.csv`                |
| 3  | `python -m src.run_feature_selection`   | Selects the Top 20 non-temporal, highly predictive features.   | Saves preprocessing pipeline & feature list. |
| 4  | `python -m src.run_xgboost_baseline`    | Trains XGBoost regressor (base model).                         | `xgboost_baseline_model_final.pkl`           |
| 5  | `python -m src.competitor_intelligence` | Generates mock competitor data & market analysis.              | Competitive analysis outputs.                |
| 6  | `python -m src.price_optimizer`         | Logic for calculating final optimal price.                     | Optimal price results.                       |
| 7  | `python -m src.dynamic_pricing_engine`  | Tests engine & returns pricing + market position.              | Example: $78.44 → "Budget Leader"            |
| 8  | `python -m src.shap_explainer`          | SHAP explainability module.                                    | Transparent SHAP output.                     |
| 9  | `python -m src.feature_mapper`          | Maps user inputs into 20 model features.                       | Successful feature transformation            |
| 10 | `python -m src.llm_recommender`         | LLM generates business insights.                               | Natural-language recommendations             |
| 11 | `python -m src.pricing_pipeline`        | Full end-to-end pricing workflow.                              | Final pricing recommendation                 |

---

## 4. Launch the Web Dashboard

Run the final command to launch the Streamlit UI.

```bash
streamlit run app/streamlit_app.py
```

---

## License

This project is intended for academic and research purposes as part of a Master's Capstone project and is released under the **MIT License**.


