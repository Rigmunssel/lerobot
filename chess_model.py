from ultralytics import YOLO

# Load your model and run inference
model = YOLO("https://huggingface.co/KanisornPutta/chess-model-yolov8m/resolve/main/chess-model-yolov8m.pt")
results = model("image_1.png")

# Grab the first result (since you only passed one image)
results[0].show()