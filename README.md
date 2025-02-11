# Groundwater Level (GWL) Prediction using GRACE and Machine Learning

## Overview
This repository contains the code and workflow for predicting **Groundwater Level (GWL)** using **GRACE satellite data** and **Machine Learning (ML) models**. The project focuses on utilizing GRACE-derived Total Water Storage Anomalies (TWSA) as an input for trained ML models to estimate GWL changes across a region. The methods developed here were presented at the **CoDS Conference**, highlighting the effectiveness of data-driven approaches in hydrological predictions.

## Key Achievements
- **Preprocessing of GRACE Data**: Extracted and masked relevant geospatial data from GRACE satellite observations.
- **Machine Learning Model Training**: Developed and trained models, including **Linear Regression, Random Forest, and XGBoost**, to predict GWL based on GRACE data.
- **Geospatial Data Handling**: Implemented efficient spatial filtering and location-based extraction of data.
- **Conference Presentation**: The methodologies and results from this work were successfully presented at **CoDS Conference**.

## How to Use This Repository
This repository provides a structured workflow to process GRACE data, apply necessary geospatial transformations, and train machine learning models for GWL prediction. The scripts included serve the following purposes:

### 1. Extract and Process Data
#### `get_data_at_a_location_tiff.py`
- Extracts GRACE-based TWSA data at specific geographic locations.
- Reads **GeoTIFF** files and retrieves time-series data for given latitudes and longitudes.
- Output: Structured dataset ready for analysis.

#### `mask_tiff.py`
- Applies spatial masking to GRACE data to focus only on the study region.
- Uses shapefiles to remove unwanted geographic areas.
- Output: Masked geospatial dataset.

### 2. Train Machine Learning Models
#### `train_Linear_Regression.py`
- Trains a **Linear Regression Model** using GRACE data as input.
- Evaluates model performance based on historical GWL records.

#### `train_Random_Forest.py`
- Implements **Random Forest Regressor** for GWL prediction.
- Utilizes multiple decision trees to improve prediction accuracy.

#### `train_XGboost.py`
- Trains an **XGBoost Model**, an advanced gradient boosting technique, for predicting GWL.
- Optimized for performance with hyperparameter tuning.

## Requirements
Ensure you have the following dependencies installed before running the scripts:
```bash
pip install numpy pandas rasterio geopandas xgboost scikit-learn matplotlib
```

## Running the Scripts
1. **Prepare the Data**
   - Use `get_data_at_a_location_tiff.py` and `mask_tiff.py` to process GRACE data.

2. **Train ML Models**
   - Run `train_Linear_Regression.py`, `train_Random_Forest.py`, or `train_XGboost.py` to train respective models.

3. **Evaluate and Use Predictions**
   - Analyze model performance and predict groundwater level trends based on GRACE inputs.

## Future Work
- **Enhancing Model Accuracy**: Incorporating additional hydrological datasets such as precipitation, soil moisture, and temperature.
- **Expanding Geospatial Coverage**: Applying the approach to different regions for broader hydrological analysis.

For any queries or contributions, feel free to reach out!

