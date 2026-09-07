from ultralytics import YOLO

if __name__ == "__main__":

    # 1. Load your trained model
    model = YOLO("runs/detect/robbery_model/initial_training-2/weights/best.pt")

    # 2. Run inference on a new image
    results = model.predict(source="dataset/valid/images/burg10_0111_jpg.rf.7e94ddee540b7f3536c490e4f9edaab5.jpg", save=True, conf=0.3)

    # 3. View the results
    for r in results:
        r.show()