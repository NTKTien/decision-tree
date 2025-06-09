# main.py
import pandas as pd
from preprocess import preprocess, split_features_labels
from pandas import pandas as pd
from utils import train_decision_tree, visualize_decision_tree, evaluate_model, analyze_max_depth

# Cấu hình
DATA_PATH = "d:/Python/decision-tree/datasets/heart_disease.csv"  # Điều chỉnh đường dẫn nếu cần
DATASET_NAME = "heart_disease"
LABEL_COL = "target"
SPLIT_RATIO = 0.8  # 80/20
CLASS_NAMES = ["No Disease", "Heart Disease"]

# Tiền xử lý dữ liệu
df = preprocess(DATA_PATH, missing_strategy='mean', categorical_strategy='label')
X, y = split_features_labels(df, LABEL_COL)

# Chia dữ liệu
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT_RATIO, stratify=y, random_state=42)

# Huấn luyện mô hình
clf = train_decision_tree(X_train, y_train, max_depth=3)

# Trực quan hóa cây quyết định
visualize_decision_tree(clf, X.columns, CLASS_NAMES, "output/heart_disease_tree")

# Đánh giá mô hình
results = evaluate_model(clf, X_test, y_test, DATASET_NAME, "80_20")
print("Độ chính xác:", results['accuracy'])

# Phân tích độ sâu tối đa
analyze_max_depth(X_train, y_train, X_test, y_test, X.columns, CLASS_NAMES, DATASET_NAME)