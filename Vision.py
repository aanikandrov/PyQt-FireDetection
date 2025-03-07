import torch
from PyQt5.QtWidgets import QGraphicsScene

from ultralytics import YOLO

import ImagesWork


def training():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print("Используется GPU")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Используется CPU (GPU не доступен)")

    model = YOLO('yolo11n.pt').to(device)
    results = model.train(
        data='data.yaml',
        imgsz=640,
        epochs=8,
        batch=8,
        lr0=1e-4,
        dropout=0.15,
        name='model',
        device=device
    )


def pred_photo(sample_path):
    model_path = 'best.pt'
    model = YOLO(model_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model.to(device)

    results = model.predict(source=sample_path,
                            imgsz=640)

    scene = QGraphicsScene()

    for result in results:
        # Как оптимизировать?
        filename = '1.jpg'
        result.save(filename)
        pixmap = ImagesWork.load_image(filename)
        # os.remove(filename)
        scene.addPixmap(pixmap)
    return scene


