# Object Detection System

## Objective
This project detects vehicle license plates in images or webcam feed using a custom-trained YOLO (You Only Look Once) deep learning model. Instead of detecting generic objects, the model is fine-tuned specifically on license plate images so it can recognize plates accurately. The system is presented through a simple, interactive web interface built with Gradio.

## Required Libraries
- ultralytics
- opencv-python
- gradio
- pandas
- pillow

## Installation Steps
1. Make sure Python 3.9 or higher is installed on your system.
2. (Recommended) Create a virtual environment so the project's libraries stay separate from other projects:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install all required libraries in one step:
   ```
   pip install -r requirements.txt
   ```

## How to Run the Project

**Step 1: Train the model (only needed once)**
```
python train.py
```
This fine-tunes a pretrained YOLO model on the license plate dataset and saves the trained model inside the `runs/detect/license_plate_run/weights/` folder as `best.pt`.

**Step 2: Launch the web app**
```
python app.py
```
This starts a local web server. A link will appear in the terminal, usually:
```
Running on local URL:  http://127.0.0.1:7860
```
Open that link in your browser.

**Step 3: Use the app**
- Upload an image (or use your webcam) showing a vehicle with a visible license plate.
- Adjust the confidence threshold slider if needed.
- Click submit to see the detection result.

## Expected Output
- An annotated image showing a bounding box drawn around each detected license plate.
- A table listing each detection with its label and confidence percentage.
- A short summary line showing the total number of license plates detected.

## Notes
- The first time you run the project, the base YOLO model file will download automatically.
- Detection accuracy depends on image quality, lighting, and how visible the license plate is in the photo.
