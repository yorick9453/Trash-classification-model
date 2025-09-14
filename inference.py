from flask import Flask, request, jsonify, render_template_string
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import io
import base64

app = Flask(__name__)

# -------------------------------
# 模型設定
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11
MODEL_PATH = "./runs/Drop_SplitTrain/best_resnet50_epoch33_acc0.9732.pth"

# 載入模型
model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, NUM_CLASSES))
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()


idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -------------------------------
# 前端 HTML 頁面
# -------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>即時攝像頭分類</title>
</head>
<body>
    <h2>即時攝像頭拍照分類</h2>
    <video id="video" width="320" height="240" autoplay></video>
    <button id="snap">拍照分類</button>
    <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>
    <p id="result"></p>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');

        // 啟動攝像頭
        navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => { video.srcObject = stream; })
        .catch(err => { console.error("無法啟動攝像頭:", err); });

        // 拍照按鈕
        document.getElementById('snap').addEventListener('click', async () => {
            context.drawImage(video, 0, 0, 320, 240);
            const dataURL = canvas.toDataURL('image/png');

            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataURL })
            });

            const data = await response.json();
            document.getElementById('result').innerText = "分類結果: " + data.prediction;
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

# -------------------------------
# API: 圖片推理
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # base64 decode
    img_bytes = base64.b64decode(data["image"].split(",")[-1])
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        class_id = pred.item()
        class_name = idx_to_class[class_id]

    return jsonify({"prediction": class_name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
