import os
import numpy as np
from sklearn.datasets import fetch_openml
from PIL import Image, ImageOps

def generate_sample_images():
    # Tạo thư mục chứa ảnh mẫu
    output_dir = "sample_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("⏳ Đang tải dữ liệu Fashion MNIST (để lấy mẫu)...")
    # Tải dữ liệu
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='auto')

    # Ép kiểu nhãn về số nguyên
    y = y.astype(np.uint8)

    # Danh sách nhãn
    class_names = {
        0: 'T-shirt_top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
        5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle_boot'
    }

    print("📸 Đang trích xuất và lưu ảnh...")

    # Duyệt qua từng class (0-9) để lấy 1 mẫu đại diện
    for label_id, label_name in class_names.items():
        # Tìm vị trí đầu tiên xuất hiện class này trong tập dữ liệu
        idx = np.where(y == label_id)[0][0]

        # Lấy vector ảnh và reshape về 28x28
        img_data = X[idx].reshape(28, 28).astype(np.uint8)

        # Tạo ảnh PIL từ mảng numpy
        img = Image.fromarray(img_data)

        # --- QUAN TRỌNG: MÔ PHỎNG ẢNH THỰC TẾ ---
        # Fashion MNIST gốc là: Nền đen (0), Vật thể trắng (255).
        # Ảnh chụp thực tế thường là: Nền trắng, Vật thể tối màu.
        # -> Ta sẽ ĐẢO MÀU (Invert) ảnh mẫu này để test xem App của bạn
        # có tự động xử lý được ảnh nền trắng không.
        img_inverted = ImageOps.invert(img)

        # Lưu file
        filename = f"{output_dir}/{label_id}_{label_name}.png"
        img_inverted.save(filename)
        print(f"   ✅ Đã lưu: {filename}")

    print(f"\n🎉 Hoàn tất! Kiểm tra thư mục '{output_dir}' để lấy ảnh test.")

if __name__ == "__main__":
    generate_sample_images()