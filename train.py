import argparse
import os

import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Tạo thư mục mlruns nếu chưa có để tránh lỗi permission cục bộ
if not os.path.exists("mlruns"):
    os.makedirs("mlruns")

mlflow.set_tracking_uri("file:./mlruns")

# Thiết lập tham số đầu vào để Tuning
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=50)
parser.add_argument("--max_depth", type=int, default=10)
args = parser.parse_args()

# Tên Experiment
mlflow.set_experiment("Fashion_MNIST_Experiment")

with mlflow.start_run():
    print("⏳ Đang tải dữ liệu Fashion MNIST...")
    # Tải dữ liệu (dạng vector 784 chiều cho mỗi ảnh)
    X, y = fetch_openml(
        "Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="auto"
    )

    # Lấy mẫu nhỏ nếu cần test nhanh (bỏ comment 2 dòng dưới nếu máy yếu)
    # X = X[:2000]
    # y = y[:2000]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Log tham số
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Huấn luyện
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
    )
    clf.fit(X_train, y_train)

    # Đánh giá
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"✅ Accuracy: {acc}")
    mlflow.log_metric("accuracy", acc)

    # Lưu model
    mlflow.sklearn.log_model(clf, "model")
