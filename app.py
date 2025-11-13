from flask import Flask, request, render_template
import mlflow.sklearn
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)

# Load model
try:
    # Load model từ thư mục local (được tạo bởi script get_best_model.py)
    model = mlflow.sklearn.load_model("./model_best/model")
    print("✅ Đã load model thành công.")
except Exception:
    print("⚠️ Chưa tìm thấy model. Hãy chạy get_best_model.py trước.")
    model = None

class_names = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}

def preprocess_image(image_file):
    """
    Hàm xử lý ảnh đầu vào:
    1. Chuyển Grayscale.
    2. Đảo màu (nếu ảnh là nền trắng vật tối).
    3. Resize giữ tỷ lệ sao cho cạnh lớn nhất = 28 (vừa khít khung).
    4. Dán vào giữa nền đen 28x28.
    """
    # 1. Mở ảnh và chuyển sang Grayscale (L)
    img = Image.open(image_file).convert("L")

    # 2. Đảo ngược màu (Invert) nếu phát hiện nền trắng
    if img.getpixel((0, 0)) > 127:
        img = ImageOps.invert(img)

    # 3. Resize giữ tỷ lệ (Aspect Ratio)
    img.thumbnail((28, 28), Image.Resampling.LANCZOS)

    # Tạo một canvas màu đen kích thước 28x28 chuẩn
    new_img = Image.new("L", (28, 28), color=0)

    # Tính toán vị trí để dán ảnh vào chính giữa
    x_offset = (28 - img.width) // 2
    y_offset = (28 - img.height) // 2
    new_img.paste(img, (x_offset, y_offset))

    new_img.save("debug_processed.png")

    # 4. Chuyển thành mảng numpy và flatten (1, 784)
    img_data = np.array(new_img).reshape(1, -1)

    return img_data

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    if request.method == "POST" and model:
        file = request.files["file"]
        if file:
            try:
                # Gọi hàm tiền xử lý
                img_data = preprocess_image(file)

                # Dự đoán
                pred = model.predict(img_data)[0]
                pred_idx = int(pred)

                prediction_text = (
                    f"Kết quả: {class_names.get(pred_idx, 'Unknown')} (ID: {pred_idx})"
                )
            except Exception as e:
                # In lỗi ra để debug nếu vẫn bị
                print(f"Error: {e}")
                prediction_text = f"Lỗi xử lý: {str(e)}"

    return render_template("index.html", prediction=prediction_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)