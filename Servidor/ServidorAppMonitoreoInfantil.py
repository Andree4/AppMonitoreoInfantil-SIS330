import torch
import flask
from flask import request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room
import logging
import os
import cv2
import numpy as np
from collections import deque
from transformers import AutoImageProcessor, TimesformerForVideoClassification, BertTokenizer, BertForSequenceClassification
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import json
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

# Initialize Flask app
app = flask.Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("server_detection.log"),
        logging.StreamHandler()
    ]
)

# Define parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FRAMES = 8
FRAME_BUFFER = deque(maxlen=NUM_FRAMES)
MODEL_PATH_TIMESFORMER = r"C:\Tareas Hechas\Inteligencia Artificial 3\MODELOS\Modelo_Detector_Violencia.pt"
MODEL_PATH_BERT = r"C:\Tareas Hechas\Inteligencia Artificial 3\MODELOS\Modelo_Bert_Texto_Ofensivo.pth"
MODEL_PATH_NSFW = r"C:\Tareas Hechas\Inteligencia Artificial 3\MODELOS\Modelo_MobileNetV2_Detector_NSFW.pt"

# Function to check collisions between bounding boxes


def check_collision(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2)


# Load YOLO model
try:
    yolo_model = YOLO("yolov8n.pt")
    logging.info("Modelo YOLO cargado correctamente")
except Exception as e:
    logging.error(f"Error cargando modelo YOLO: {e}")
    raise

# Load Timesformer model and processor
try:
    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/timesformer-base-finetuned-k400")
    timesformer_model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400", num_labels=2, ignore_mismatched_sizes=True
    )
    state_dict = torch.load(MODEL_PATH_TIMESFORMER, map_location=DEVICE)
    timesformer_model.load_state_dict(state_dict)
    timesformer_model.to(DEVICE)
    timesformer_model.eval()
    logging.info(f"Modelo Timesformer cargado desde {MODEL_PATH_TIMESFORMER}")
except Exception as e:
    logging.error(f"Error cargando modelo Timesformer: {e}")
    raise

# Load BERT model and tokenizer
try:
    bert_model = BertForSequenceClassification.from_pretrained(
        'dccuchile/bert-base-spanish-wwm-uncased', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(
        'dccuchile/bert-base-spanish-wwm-uncased')
    if os.path.exists(MODEL_PATH_BERT):
        bert_model.load_state_dict(torch.load(
            MODEL_PATH_BERT, map_location=DEVICE))
        logging.info(f"Modelo BERT cargado desde {MODEL_PATH_BERT}")
    else:
        logging.warning(
            f"Archivo de modelo BERT {MODEL_PATH_BERT} no encontrado. Usando modelo preentrenado.")
    bert_model.to(DEVICE)
    bert_model.eval()
except Exception as e:
    logging.error(f"Error cargando modelo BERT: {e}")
    raise

# Load NSFW model (MobileNetV2)
try:
    nsfw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    nsfw_model = models.mobilenet_v2(pretrained=False)
    nsfw_model.classifier[1] = nn.Sequential(
        nn.Linear(nsfw_model.last_channel, 1),
        nn.Sigmoid()
    )
    nsfw_model.load_state_dict(torch.load(
        MODEL_PATH_NSFW, map_location=DEVICE))
    nsfw_model.to(DEVICE)
    nsfw_model.eval()
    logging.info(f"Modelo NSFW (MobileNetV2) cargado desde {MODEL_PATH_NSFW}")
except Exception as e:
    logging.error(f"Error cargando modelo NSFW: {e}")
    raise

# Function to predict offensive text


def predict_text(text):
    if "*" in text:
        logging.debug(
            "Automatic classification as offensive due to '*' in text: %s", text)
        return "Ofensivo"
    inputs = tokenizer(text, return_tensors="pt", padding=True,
                       truncation=True, max_length=10).to(DEVICE)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "Ofensivo" if predicted_class == 1 else "No ofensivo"

# Function to predict NSFW content


def predict_nsfw(image):
    try:
        input_tensor = nsfw_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = nsfw_model(input_tensor)
            prob = output.item()
            label = "No apto" if prob > 0.5 else "Apto"
        return label, prob
    except Exception as e:
        logging.error(f"Error en inferencia NSFW: {e}")
        return "Error", 0.0

# Function to process image for violence and NSFW detection


def process_image_for_violence_and_nsfw(image_data):
    try:
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        frame_rgb = np.array(img)

        FRAME_BUFFER.append(frame_rgb)

        # Violence detection
        collision_detected = False
        results = yolo_model(frame_rgb, classes=[0])
        boxes = results[0].boxes.xywh.cpu().numpy()
        if len(boxes) > 1:
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if check_collision(boxes[i], boxes[j]):
                        collision_detected = True
                        break
                if collision_detected:
                    break

        violence_label = "No Violence"
        if len(FRAME_BUFFER) == NUM_FRAMES and collision_detected:
            try:
                inputs = image_processor(
                    list(FRAME_BUFFER), return_tensors="pt")
                video_tensor = inputs["pixel_values"].to(DEVICE)
                with torch.no_grad():
                    outputs = timesformer_model(pixel_values=video_tensor)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
                violence_label = "Violence" if pred == 1 else "No Violence"
            except Exception as e:
                logging.warning(f"Error en inferencia Timesformer: {e}")
                violence_label = "Error"

        # NSFW detection
        nsfw_label, nsfw_prob = predict_nsfw(img)
        logging.debug(
            f"NSFW classification: {nsfw_label} (prob: {nsfw_prob:.4f})")

        return violence_label, nsfw_label, boxes
    except Exception as e:
        logging.error(f"Error procesando imagen: {e}")
        return "Error", "Error", []

# Route to verify connection


@app.route('/', methods=['GET'])
def handle_call():
    return "Successfully Connected"

# Route for testing


@app.route('/getfact', methods=['GET'])
def get_fact():
    return "Hey!! I'm the fact you got!!!"

# SocketIO handlers for text namespace


@socketio.on('connect', namespace='/text')
def handle_text_connect():
    logging.debug("Client connected to /text namespace")
    join_room('text_room')


@socketio.on('transcription', namespace='/text')
def handle_transcription(data):
    try:
        logging.debug("Received transcription data: %s", data)
        if isinstance(data, str):
            data = json.loads(data)
        text = data.get('text', '')
        logging.debug("Parsed transcription text: %s", text)
        if not text.strip():
            logging.warning("No text provided")
            emit('error', {'message': 'No se proporcionó texto'},
                 namespace='/text')
            return

        result = predict_text(text)
        logging.debug("Classification result: %s for text: %s", result, text)
        emit('classification', {'text': text,
             'classification': result}, namespace='/text')
        if result == "Ofensivo":
            emit('offensive_notification', {
                 'text': text, 'classification': result}, namespace='/text', room='text_room')
    except json.JSONDecodeError as e:
        logging.error(
            f"Error decoding JSON in transcription: {e}, data: %s", data)
        emit('error', {'message': 'Invalid JSON format'}, namespace='/text')
    except Exception as e:
        logging.error(f"Error processing transcription: {e}, data: %s", data)
        emit('error', {'message': 'Server error'}, namespace='/text')

# SocketIO handlers for video namespace


@socketio.on('connect', namespace='/video')
def handle_video_connect():
    logging.debug("Client connected to /video namespace")
    join_room('video_room')


@socketio.on('image', namespace='/video')
def handle_image(data):
    try:
        logging.debug("Received image data: %s", data)
        if isinstance(data, str):
            data = json.loads(data)
        image_data = data.get('image', '')
        logging.debug("Parsed image data length: %d", len(image_data))
        if not image_data:
            logging.warning("No image provided")
            emit('error', {'message': 'No se proporcionó imagen'},
                 namespace='/video')
            return

        violence_label, nsfw_label, boxes = process_image_for_violence_and_nsfw(
            image_data)
        logging.debug(
            f"Violence detection result: {violence_label}, NSFW detection result: {nsfw_label}")
        emit('violence_classification', {
             'violence_classification': violence_label,
             'nsfw_classification': nsfw_label
             }, namespace='/video')
        if violence_label == "Violence":
            emit('violence_notification', {
                 'classification': violence_label}, namespace='/video', room='video_room')
        if nsfw_label == "No apto":
            emit('nsfw_notification', {
                 'classification': nsfw_label}, namespace='/video', room='video_room')
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in image: {e}, data: %s", data)
        emit('error', {'message': 'Invalid JSON format'}, namespace='/video')
    except Exception as e:
        logging.error(f"Error processing image: {e}, data: %s", data)
        emit('error', {'message': 'Server error'}, namespace='/video')


if __name__ == "__main__":
    import eventlet
    eventlet.monkey_patch()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
