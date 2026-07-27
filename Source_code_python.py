''' from ultralytics import YOLO

# Train the model
model = YOLO("yolo26n.pt")

model.train(
    data="Dataset/data.yaml",  
    epochs=30,
    imgsz=640,
    patience=10,   
    name="license_plate_run"
) '''

import gradio as gr
from ultralytics import YOLO
import pandas as pd
from PIL import Image


model = YOLO('runs/detect/license_plate_run/weights/best.pt')

def detect(image, confidence):
    # img = Image.fromarray(image)
    results = model(image, conf=confidence)
    annotated = results[0].plot()
    data = []
    for box in results[0].boxes:
        cls = model.names[int(box.cls)]
        conf = float(box.conf)
        data.append([cls, f"{conf:.2%}"])
    df = pd.DataFrame(data, columns=["Object", "Confidence"])
    count_text = f"Total objects detected: {len(data)}"
    return annotated, df, count_text

demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="Upload Image or Use Webcam", sources=["upload", "webcam"]),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.25, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(type="numpy", label="Detected Objects"),
        gr.Dataframe(label="Detection Details"),
        gr.Textbox(label="Summary")
    ],
    title="License Plate Detector",
    description="Upload an image or use your webcam to detect license plates using a custom-trained YOLO model."
)
demo.launch()
