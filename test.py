
from PIL import Image

# YOLOv5モデルを読み込む
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 推論を行う画像を読み込む
img = Image.open('nogi.jpg')

# 画像から物体検出を行う
results = model(img)

# 検出結果を表示する
results.show()

# 検出結果を保存する
results.save('result.jpg')
