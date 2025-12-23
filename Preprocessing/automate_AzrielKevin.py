import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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

def preprocess_data(input_path, output_path, target_column="traffic_volume"):
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

    # 3️⃣ Cleaning dasar
    df = replace_out_of_range_with_median(df, valid_ranges)
    df = clean_categorical_columns(df, categorical_cols)
    df = convert_comma_to_float(df, float_cols)
    df = handle_outlier_iqr_capping(df, "temp")
    df = df.drop_duplicates()

    # 4️⃣ Tangani missing values
    # Numerik → median
    num_cols = ["temp", "rain_1h", "snow_1h", "clouds_all", target_column]
    for col in num_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    # Kategorikal → "unknown"
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna("unknown", inplace=True)

    # 5️⃣ One-hot encoding kategorikal
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # 6️⃣ Scaling numerik (kecuali target)
    scale_cols = ["temp", "rain_1h", "snow_1h", "clouds_all"]
    scale_cols = [c for c in scale_cols if c in df.columns and c != target_column]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # 7️⃣ Simpan hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Preprocessing selesai, hasil disimpan di: {output_path}")
    return df

# =========================
# CLI
# =========================

if __name__ == "__main__":
    INPUT_PATH = "Dataset/Metro_Interstate_Traffic_Volume.csv"
    OUTPUT_PATH = "Preprocessing/Metro_Interstate_Traffic_Volume_preprocessing.csv"
    TARGET_COLUMN = "traffic_volume"

    preprocess_data(INPUT_PATH, OUTPUT_PATH, TARGET_COLUMN)
