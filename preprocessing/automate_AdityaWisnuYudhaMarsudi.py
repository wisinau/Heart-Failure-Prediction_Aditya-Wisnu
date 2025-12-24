import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(path):
    # Cek apakah file ada sebelum membaca
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan di: {path}")
    return pd.read_csv(path)

def preprocess_data(df):
    # Hapus duplikat
    df = df.drop_duplicates()

    # Pisahkan Fitur dan Target
    target = 'HeartDisease'
    X = df.drop(target, axis=1)
    y = df[target]

    # Definisi kolom
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    # Buat Pipeline Preprocessing
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Fit & Transform
    X_processed = preprocessor.fit_transform(X)

    # Ambil nama kolom baru
    new_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_cols = numerical_cols + list(new_cat_cols)

    # Gabungkan kembali
    X_df = pd.DataFrame(X_processed, columns=all_cols)
    final_df = pd.concat([X_df, y.reset_index(drop=True)], axis=1)

    return final_df

if __name__ == "__main__":
    # --- LOGIKA PENENTUAN PATH OTOMATIS ---
    # Kita cek beberapa kemungkinan lokasi file agar script tidak error
    possible_paths = [
        "heart_failure_raw/heart.csv",       # Lokasi default di Colab root
        "../heart_failure_raw/heart.csv",    # Lokasi sesuai struktur folder Submission (jika dijalankan dari folder preprocessing)
        "/content/heart_failure_raw/heart.csv" # Lokasi absolut Colab
    ]

    input_path = None
    for path in possible_paths:
        if os.path.exists(path):
            input_path = path
            break

    # Jika file tetap tidak ketemu
    if input_path is None:
        print("ERROR: File 'heart.csv' tidak ditemukan di lokasi manapun.")
        print("Pastikan folder 'heart_failure_raw' sudah ada dan berisi file heart.csv.")
        exit()

    # Tentukan output path
    # Jika script dijalankan dari root, simpan ke folder preprocessing/heart_failure_preprocessing
    if "preprocessing" not in os.getcwd():
         output_path = "preprocessing/heart_failure_preprocessing/heart_clean.csv"
    else:
         output_path = "heart_failure_preprocessing/heart_clean.csv"

    # Buat folder output jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading Data dari: {input_path}")
    df = load_data(input_path)

    print("Preprocessing Data...")
    df_clean = preprocess_data(df)

    print(f"Saving Data ke: {output_path} ...")
    df_clean.to_csv(output_path, index=False)
    print("Berhasil! Data Preprocessing Selesai.")
