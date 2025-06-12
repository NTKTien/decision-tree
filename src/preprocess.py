import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

def preprocess(
    filepath, 
    missing_strategy='drop',  # 'drop', 'mean', 'median'
    categorical_strategy='label'  # 'label' or 'onehot'
):
    df = pd.read_csv(filepath)

    # Xử lý giá trị thiếu
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy == 'mean':
        for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col] = df[col].fillna(df[col].mean())
        df = df.dropna()
    elif missing_strategy == 'median':
        for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col] = df[col].fillna(df[col].median())
        df = df.dropna()

    # Xử lý categorical
    cat_cols = df.select_dtypes(include=['object']).columns
    if categorical_strategy == 'label':
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    elif categorical_strategy == 'onehot':
        df = pd.get_dummies(df, columns=cat_cols)

    return df

def split_features_labels(df, label_col):
    """
    Tách đặc trưng (X) và nhãn (y) từ DataFrame.
    """
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y

def plot_label_distribution(y, title):
    """
    Vẽ biểu đồ phân phối lớp cho nhãn y với tiêu đề title.
    """
    plt.figure(figsize=(4,3))
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def get_dataset_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    dataset_dir = os.path.abspath(os.path.join(base_dir, "..", "datasets"))
    return os.path.join(dataset_dir, filename)

if __name__ == "__main__":
    datasets = {
        "palmer_penguins": {
            "path": get_dataset_path("palmer_penguins.csv"),
            "label": "species"
        },
        "heart_disease": {
            "path": get_dataset_path("heart_disease.csv"),
            "label": "target"
        },
    }
    split_ratios = [
        (0.4, 0.6),
        (0.6, 0.4),
        (0.8, 0.2),
        (0.9, 0.1)
    ]
    results = {}

    for ds_name, ds_info in datasets.items():
        print(f"\n=== Dataset: {ds_name} ===")
        df = preprocess(ds_info["path"], missing_strategy='mean', categorical_strategy='label')
        X, y = split_features_labels(df, ds_info["label"])

        # Vẽ phân phối lớp của tập gốc
        plot_label_distribution(y, f"{ds_name} - Original label distribution")

        for train_ratio, test_ratio in split_ratios:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                train_size=train_ratio, 
                test_size=test_ratio, 
                stratify=y, 
                random_state=42
            )
            train_pct = int(train_ratio * 100)
            test_pct = int(test_ratio * 100)
            # Tạo biến toàn cục cho từng tập
            globals()[f"feature_train_{train_pct}"] = X_train
            globals()[f"label_train_{train_pct}"] = y_train
            globals()[f"feature_test_{test_pct}"] = X_test
            globals()[f"label_test_{test_pct}"] = y_test

            key = f"{ds_name}_{train_pct}_{test_pct}"
            results[key] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test
            }
            print(f"Split {train_pct}% train - {test_pct}% test: "
                  f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, "
                  f"Train label dist: {y_train.value_counts(normalize=True).to_dict()}, "
                  f"Test label dist: {y_test.value_counts(normalize=True).to_dict()}")

            # Vẽ phân phối lớp cho tập train
            plot_label_distribution(y_train, f"{ds_name} - Train {train_pct}% label distribution")

            # Vẽ phân phối lớp cho tập test
            plot_label_distribution(y_test, f"{ds_name} - Test {test_pct}% label distribution")
