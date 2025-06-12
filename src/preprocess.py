import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_dataset_path(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    dataset_dir = os.path.abspath(os.path.join(base_dir, "..", "datasets"))
    return os.path.join(dataset_dir, file_name)


def load_and_preprocess_dataset(file_name, label_column='label'):
    df = pd.read_csv(get_dataset_path(file_name))
    # Sử dụng ascii hoặc loại bỏ ký tự unicode khi in ra terminal
    print("Du lieu ban dau:", df.shape)

    # Xử lý missing value (giản lược: loại bỏ các dòng thiếu)
    df = df.dropna()

    # Encode các cột object (bao gồm nhãn nếu cần)
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=[label_column])
    y = df[label_column]
    return X, y, df, encoders


def prepare_all_splits(X, y, train_ratios=[40, 60, 80, 90]):
    """
    Chuẩn bị các tập train/test với các tỷ lệ khác nhau, shuffle và stratify.
    Trả về dict chứa các tập và nhãn gốc.
    """
    # Shuffle trước khi chia
    X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
    splits = {}
    for ratio in train_ratios:
        train_size = ratio / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X_shuffled, y_shuffled,
            train_size=train_size,
            stratify=y_shuffled,
            random_state=42,
            shuffle=True
        )
        splits[ratio] = {
            'feature_train': X_train,
            'label_train': y_train,
            'feature_test': X_test,
            'label_test': y_test
        }
        # Tạo biến toàn cục cho từng tập
        globals()[f"feature_train_{ratio}"] = X_train
        globals()[f"label_train_{ratio}"] = y_train
        globals()[f"feature_test_{100-ratio}"] = X_test
        globals()[f"label_test_{100-ratio}"] = y_test
    return splits, y_shuffled


def plot_class_distributions(original_labels, splits, save=False):
    """
    Vẽ biểu đồ phân phối nhãn cho tập gốc, train, test của từng tỷ lệ.
    """
    ratios = list(splits.keys())
    num_plots = 1 + 2 * len(ratios)
    rows = (num_plots + 2) // 3
    plt.figure(figsize=(15, rows * 4))

    # Plot original distribution
    plt.subplot(rows, 3, 1)
    sns.countplot(x=original_labels)
    plt.title("Original Dataset")

    plot_index = 2
    for ratio in ratios:
        train_labels = splits[ratio]['label_train']
        test_labels = splits[ratio]['label_test']

        plt.subplot(rows, 3, plot_index)
        sns.countplot(x=train_labels)
        plt.title(f"Train ({ratio}%)")
        plot_index += 1

        plt.subplot(rows, 3, plot_index)
        sns.countplot(x=test_labels)
        plt.title(f"Test ({100 - ratio}%)")
        plot_index += 1

    plt.tight_layout()
    if save:
        plt.savefig("class_distributions.png")
    plt.show()


def main():
    # Ví dụ với dữ liệu heart_disease
    X, y, df, encoders = load_and_preprocess_dataset("heart_disease.csv", label_column="target")
    print("Shape X:", X.shape, "Shape y:", y.shape)
    splits, y_shuffled = prepare_all_splits(X, y)
    plot_class_distributions(y_shuffled, splits)

    # Ví dụ với dữ liệu palmer_penguins
    X2, y2, df2, encoders2 = load_and_preprocess_dataset("palmer_penguins.csv", label_column="species")
    print("Shape X2:", X2.shape, "Shape y2:", y2.shape)
    splits2, y2_shuffled = prepare_all_splits(X2, y2)
    plot_class_distributions(y2_shuffled, splits2)

if __name__ == "__main__":
    main()
