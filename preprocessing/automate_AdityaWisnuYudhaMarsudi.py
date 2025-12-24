import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- FUNGSI UTAMA ---
def process_heart_data():
    # 1. Tentukan Lokasi File secara Absolut (Anti-Gagal)
    # Lokasi script ini berada (folder 'preprocessing')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lokasi root repository (naik satu level dari script_dir)
    repo_root = os.path.dirname(script_dir)
    
    # Path Input (Raw Data)
    input_path = os.path.join(repo_root, 'heart_failure_raw', 'heart.csv')
    
    # Path Output (Clean Data)
    output_folder = os.path.join(script_dir, 'heart_failure_preprocessing')
    output_path = os.path.join(output_folder, 'heart_clean.csv')

    print(f"[INFO] Mencari data di: {input_path}")

    # 2. Cek Keberadaan File
    if not os.path.exists(input_path):
        # Fallback: Coba cek jika script dijalankan langsung dari root (kasus Colab tertentu)
        input_path_alt = "heart_failure_raw/heart.csv"
        if os.path.exists(input_path_alt):
             input_path = input_path_alt
        else:
            print(f"[ERROR] File tidak ditemukan di: {input_path}")
            print("Pastikan folder 'heart_failure_raw' dan file 'heart.csv' ada di repo!")
            sys.exit(1) # Hentikan program dengan kode error

    # 3. Load & Preprocessing
    try:
        df = pd.read_csv(input_path)
        print(f"[INFO] Data dimuat. Dimensi awal: {df.shape}")

        # Hapus Duplikat
        df = df.drop_duplicates()

        # Pisahkan Target
        target = 'HeartDisease'
        X = df.drop(target, axis=1)
        y = df[target]

        # Pipeline Preprocessing
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ]
        )

        X_processed = preprocessor.fit_transform(X)
        
        # Ambil nama kolom baru
        new_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        all_cols = numerical_cols + list(new_cat_cols)
        
        # Gabungkan hasil
        X_df = pd.DataFrame(X_processed, columns=all_cols)
        final_df = pd.concat([X_df, y.reset_index(drop=True)], axis=1)
        
        # 4. Simpan Hasil
        os.makedirs(output_folder, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"[SUKSES] Data bersih disimpan di: {output_path}")

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    process_heart_data()
