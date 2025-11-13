import mlflow
import shutil
import os


def get_best_model():
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "Fashion_MNIST_Experiment"

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print("❌ Không tìm thấy Experiment.")
        return

    # Tìm run có Accuracy cao nhất
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )

    if len(runs) == 0:
        print("❌ Chưa có lần chạy nào.")
        return

    best_run = runs.iloc[0]
    best_run_id = best_run.run_id
    best_acc = best_run["metrics.accuracy"]

    print(f"🏆 Best Run ID: {best_run_id} | Accuracy: {best_acc}")

    # Tải model về thư mục cục bộ để đóng gói vào Docker
    artifact_path = "model"
    local_path = "./model_best"

    if os.path.exists(local_path):
        shutil.rmtree(local_path)

    mlflow.artifacts.download_artifacts(
        run_id=best_run_id, artifact_path=artifact_path, dst_path=local_path
    )
    print(f"💾 Đã lưu model tốt nhất tại: {local_path}")


if __name__ == "__main__":
    get_best_model()
