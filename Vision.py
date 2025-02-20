
from PyQt5.QtWidgets import QGraphicsScene

from ultralytics import YOLO

import ImagesWork


def training():
    model = YOLO("yolo11n.pt")
    results = model.train(
        data='D:/Fire/data.yaml',
        imgsz=640,
        epochs=2,
        batch=8,
        name='model')


def pred_photo(sample_path):
    model_path = 'best.pt'
    model = YOLO(model_path)

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


