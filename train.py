from ultralytics import YOLO
import os
import torch

MODEL = "yolov5mu_custom_trained.pt"
DATASET_PATH = "custom_train"
EPOCHS = 25
IMG_SIZE = 640


def main():
    full_path = os.path.abspath(DATASET_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL)
    model.train(
        data=full_path + "/data.yaml",
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=device,
    )


if __name__ == "__main__":
    main()
