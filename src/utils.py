# utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import graphviz
import os

def train_decision_tree(X_train, y_train, max_depth=None, criterion='entropy', random_state=42):
    """
    Huấn luyện mô hình cây quyết định với các tham số được chỉ định.
    
    Args:
        X_train (array-like): Đặc trưng tập huấn luyện.
        y_train (array-like): Nhãn tập huấn luyện.
        max_depth (int, optional): Độ sâu tối đa của cây. Mặc định là None.
        criterion (str): Tiêu chí phân tách ('entropy' cho information gain). Mặc định là 'entropy'.
        random_state (int): Seed cho tính ngẫu nhiên. Mặc định là 42.
    
    Returns:
        DecisionTreeClassifier: Mô hình cây quyết định đã huấn luyện.
    """
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf

def visualize_decision_tree(clf, feature_names, class_names, output_path):
    """
    Trực quan hóa cây quyết định bằng Graphviz và lưu vào file.
    
    Args:
        clf (DecisionTreeClassifier): Mô hình cây quyết định đã huấn luyện.
        feature_names (list): Danh sách tên các đặc trưng.
        class_names (list): Danh sách tên các lớp.
        output_path (str): Đường dẫn lưu file hình ảnh cây quyết định.
    """
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(output_path, format='png', view=False)
    print(f"Đã lưu cây quyết định tại: {output_path}.png")

def evaluate_model(clf, X_test, y_test, dataset_name, split_ratio):
    """
    Đánh giá mô hình trên tập kiểm tra và lưu báo cáo.
    
    Args:
        clf (DecisionTreeClassifier): Mô hình cây quyết định đã huấn luyện.
        X_test (array-like): Đặc trưng tập kiểm tra.
        y_test (array-like): Nhãn tập kiểm tra.
        dataset_name (str): Tên tập dữ liệu (e.g., 'heart_disease').
        split_ratio (str): Tỷ lệ chia (e.g., '80_20').
    
    Returns:
        dict: Kết quả đánh giá bao gồm báo cáo phân loại, ma trận nhầm lẫn và độ chính xác.
    """
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    # Lưu báo cáo phân loại dưới dạng văn bản
    output_dir = f"output/report/{dataset_name}/{split_ratio}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/classification_report.txt", 'w') as f:
        f.write(classification_report(y_test, y_pred))
    
    # Vẽ và lưu ma trận nhầm lẫn
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title(f"Ma trận nhầm lẫn - {dataset_name} ({split_ratio})")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    print(f"Đã lưu báo cáo phân loại và ma trận nhầm lẫn tại: {output_dir}")
    return {'classification_report': report, 'confusion_matrix': cm, 'accuracy': acc}

def analyze_max_depth(X_train, y_train, X_test, y_test, feature_names, class_names, dataset_name, max_depths=[None, 2, 3, 4, 5, 6, 7]):
    """
    Phân tích ảnh hưởng của max_depth đến độ chính xác và trực quan hóa.
    
    Args:
        X_train (array-like): Đặc trưng tập huấn luyện.
        y_train (array-like): Nhãn tập huấn luyện.
        X_test (array-like): Đặc trưng tập kiểm tra.
        y_test (array-like): Nhãn tập kiểm tra.
        feature_names (list): Danh sách tên các đặc trưng.
        class_names (list): Danh sách tên các lớp.
        dataset_name (str): Tên tập dữ liệu.
        max_depths (list): Danh sách các giá trị max_depth để thử.
    
    Returns:
        dict: Kết quả độ chính xác cho từng max_depth.
    """
    accuracy_results = {}
    output_dir = f"output/report/{dataset_name}/max_depth_ver"
    os.makedirs(output_dir, exist_ok=True)
    
    for depth in max_depths:
        # Huấn luyện mô hình với max_depth cụ thể
        clf = train_decision_tree(X_train, y_train, max_depth=depth)
        
        # Trực quan hóa cây quyết định
        visualize_decision_tree(clf, feature_names, class_names, f"{output_dir}/tree_depth_{depth if depth is not None else 'None'}")
        
        # Đánh giá mô hình
        results = evaluate_model(clf, X_test, y_test, dataset_name, f"80_20_depth_{depth if depth is not None else 'None'}")
        accuracy_results[depth] = results['accuracy']
    
    # Vẽ biểu đồ độ chính xác theo max_depth
    depths = [str(d) if d is not None else 'None' for d in max_depths]
    accuracies = [accuracy_results[d] for d in max_depths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths, accuracies, marker='o')
    plt.title(f"Độ chính xác theo độ sâu cây - {dataset_name}")
    plt.xlabel("Độ sâu tối đa (max_depth)")
    plt.ylabel("Độ chính xác")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_report.png")
    plt.close()
    
    # Tạo bảng độ chính xác
    accuracy_table = pd.DataFrame({
        'max_depth': depths,
        'Accuracy': [f"{acc*100:.2f}%" for acc in accuracies]
    })
    accuracy_table.to_csv(f"{output_dir}/accuracy_table.csv", index=False)
    
    print(f"Đã lưu biểu đồ và bảng độ chính xác tại: {output_dir}")
    return accuracy_results

def comparative_analysis(datasets_results, output_path):
    """
    Phân tích so sánh hiệu suất giữa các tập dữ liệu.
    
    Args:
        datasets_results (dict): Kết quả độ chính xác của các tập dữ liệu.
        output_path (str): Đường dẫn lưu biểu đồ so sánh.
    """
    plt.figure(figsize=(12, 6))
    for dataset_name, results in datasets_results.items():
        depths = [str(d) if d is not None else 'None' for d in results.keys()]
        accuracies = [results[d] for d in results.keys()]
        plt.plot(depths, accuracies, marker='o', label=dataset_name)
    
    plt.title("So sánh độ chính xác theo độ sâu cây giữa các tập dữ liệu")
    plt.xlabel("Độ sâu tối đa (max_depth)")
    plt.ylabel("Độ chính xác")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Đã lưu biểu đồ so sánh tại: {output_path}")