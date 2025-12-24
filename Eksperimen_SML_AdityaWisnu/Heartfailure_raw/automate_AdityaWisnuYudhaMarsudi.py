import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def process_heart_data():
    print("="*40)
    print("   MULAI PROSES AUTOMATION   ")
    print("="*40)

    # 1. Deteksi Lokasi (Current Working Directory)
    # GitHub Actions menjalankan script dari Root Repository
    base_path = os.getcwd()
    print(f"[INFO] Posisi script berjalan di: {base_path}")

    # 2. Tentukan Path Input (Raw Data)
    # Kita cari file heart.csv di folder heart_failure_raw
    input_path = os.path.join(base_path, 'heart_failure_raw', 'heart.csv')
    print(f"[INFO] Target Input File: {input_path}")

    # 3. Tentukan Path Output (Clean Data)
    # Kita simpan di preprocessing/heart_failure_preprocessing/heart_clean.csv
    output_folder = os.path.join(base_path, 'preprocessing', 'heart_failure_preprocessing')
    output_path = os.path.join(output_folder, 'heart_clean.csv')

    # 4. Validasi Keberadaan File Input
    if not os.path.exists(input_path):
        print(f"[ERROR] File TIDAK DITEMUKAN di: {input_path}")
        print("Cek kembali: Apakah folder 'heart_failure_raw' dan file 'heart.csv' sudah ada di repo?")
        
        # Coba debugging isi folder untuk melihat apa yang ada
        print("[DEBUG] Isi folder saat ini:")
        print(os.listdir(base_path))
        sys.exit(1) # Matikan program dengan error

    # 5. Proses Data
    try:
        print("[INFO] Memuat dataset...")
        df = pd.read_csv(input_path)
        print(f"[INFO] Data dimuat. Ukuran: {df.shape}")

        # --- PREPROCESSING START ---
        # Hapus Duplikat
        df = df.drop_duplicates()

        # Pisahkan Target
        target = 'HeartDisease'
        if target not in df.columns:
            raise ValueError(f"Kolom target '{target}' tidak ditemukan di dataset!")

        X = df.drop(target, axis=1)
        y = df[target]

        # Definisi Kolom
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

        # Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
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
        # --- PREPROCESSING END ---

        # 6. Simpan Hasil
        print(f"[INFO] Membuat folder output: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"[INFO] Menyimpan file ke: {output_path}")
        final_df.to_csv(output_path, index=False)
        
        print("="*40)
        print("   SUKSES! PREPROCESSING SELESAI   ")
        print("="*40)

    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan: {e}")
        sys.exit(1)

if __name__ == "__main__":
    process_heart_data()
