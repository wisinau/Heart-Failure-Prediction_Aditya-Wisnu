import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- KONFIGURASI ---
DAGSHUB_USERNAME = "wisinau"
REPO_NAME = "Heart_Failure_Prediction_AdityaWisnuYudhaMarsudi"

def train_model():
    print("="*40)
    print("   MULAI TRAINING (CI/CD VERSION)   ")
    print("="*40)

    # 1. SETUP OTENTIKASI OTOMATIS (Tanpa Input Manual)
    # GitHub Actions akan mengirim token lewat Environment Variable
    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        print("[ERROR] Token DagsHub tidak ditemukan di Environment Variable!")
        sys.exit(1)
    
    dagshub.auth.add_app_token(token)
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("Heart Failure Prediction - CI Pipeline")

    # 2. LOAD DATA
    # Script dijalankan dari root repository oleh GitHub Actions
    base_path = os.getcwd()
    print(f"[INFO] Working Directory: {base_path}")
    
    # Prioritas Path
    possible_paths = [
        os.path.join(base_path, "heart_failure_raw", "heart.csv"),
        "heart.csv"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
            
    if not data_path:
        print(f"[ERROR] File heart.csv tidak ditemukan di: {possible_paths}")
        sys.exit(1)

    print(f"[INFO] Data ditemukan di: {data_path}")
    df = pd.read_csv(data_path)

    # 3. PREPROCESSING PIPELINE
    df = df.drop_duplicates()
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # Deteksi kolom
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # Transformasi Data
    X_processed = preprocessor.fit_transform(X)
    new_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_cols = numerical_cols + list(new_cat_cols)
    X_clean = pd.DataFrame(X_processed, columns=all_cols)

    # 4. TRAINING
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [50, 100], 'max_depth': [10], 'min_samples_split': [5]}

    with mlflow.start_run(run_name="CI_Automated_Run"):
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        print(f"[INFO] Metrics: {metrics}")

        # Logging
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")

        # Artefak (Confusion Matrix)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        print("[SUKSES] Training Selesai & Terupload ke DagsHub.")

if __name__ == "__main__":
    train_model()