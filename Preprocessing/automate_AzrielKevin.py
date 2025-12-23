import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from joblib import dump
import os

# =========================
# CLEANING FUNCTIONS
# =========================

def replace_out_of_range_with_median(data, valid_ranges):
    data = data.copy()
    for col, (min_val, max_val) in valid_ranges.items():
        valid_median = data[(data[col] >= min_val) & (data[col] <= max_val)][col].median()
        data[col] = data[col].apply(lambda x: valid_median if not (min_val <= x <= max_val) else x)
    return data

def clean_categorical_columns(data, categorical_cols):
    data = data.copy()
    data[categorical_cols] = data[categorical_cols].apply(lambda x: x.astype(str).str.strip().str.lower())
    return data

def convert_comma_to_float(data, columns):
    data = data.copy()
    for col in columns:
        data[col] = data[col].replace(',', '.', regex=True).astype(float)
    return data

def handle_outlier_iqr_capping(data, column):
    data = data.copy()
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    valid_min = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)][column].min()
    valid_max = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)][column].max()
    data[column] = data[column].apply(lambda x: valid_min if x < lower_bound else (valid_max if x > upper_bound else x))
    return data

# =========================
# MAIN PREPROCESSING FUNCTION
# =========================

def preprocess_data(input_path, output_dir, target_column="traffic_volume"):
    # 1️⃣ Baca CSV
    df = pd.read_csv(input_path)

    # 2️⃣ Tentukan valid ranges dan kolom
    valid_ranges = {
        "temp": (200, 350),
        "rain_1h": (0, 500),
        "snow_1h": (0, 500),
        "clouds_all": (0, 100),
        "traffic_volume": (0, 100000),
    }
    categorical_cols = ["weather_main", "holiday", "weather_description"]
    float_cols = ["rain_1h", "snow_1h"]

    # 3️⃣ Cleaning
    df = replace_out_of_range_with_median(df, valid_ranges)
    df = clean_categorical_columns(df, categorical_cols)
    df = convert_comma_to_float(df, float_cols)
    df = handle_outlier_iqr_capping(df, "temp")
    df = df.drop_duplicates()

    # 4️⃣ One-hot encoding kategorikal
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # 5️⃣ Scaling numerik
    numeric_features = ["temp", "rain_1h", "snow_1h", "clouds_all"]
    numeric_features = [c for c in numeric_features if c in df_encoded.columns and c != target_column]
    scaler = StandardScaler()
    df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

    # 6️⃣ Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", "passthrough")]), [c for c in df_encoded.columns if c not in numeric_features + [target_column]])
        ]
    )
    preprocessor.fit(df_encoded.drop(columns=[target_column]))

    # 7️⃣ Simpan hasil
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "Metro_Interstate_Traffic_Volume_preprocessing.csv")
    pipeline_path = os.path.join(output_dir, "preprocessor_pipeline.joblib")
    header_path = os.path.join(output_dir, "feature_header.csv")

    df_encoded.to_csv(csv_path, index=False)
    dump(preprocessor, pipeline_path)
    pd.DataFrame(columns=df_encoded.drop(columns=[target_column]).columns).to_csv(header_path, index=False)

    print(f"✅ Preprocessing selesai. File disimpan di: {csv_path}")
    return df_encoded

# =========================
# CLI
# =========================

if __name__ == "__main__":
    INPUT_PATH = "Dataset/Metro_Interstate_Traffic_Volume.csv"
    OUTPUT_DIR = "Preprocessing/Metro_Interstate_Traffic_Volume_preprocessing"
    TARGET_COLUMN = "traffic_volume"

    preprocess_data(INPUT_PATH, OUTPUT_DIR, TARGET_COLUMN)
