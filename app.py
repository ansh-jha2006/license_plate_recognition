from flask import Flask, render_template, Response, jsonify
import cv2
import pytesseract
import re
import numpy as np

app = Flask(__name__)
recognized_plates = []

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load custom cascade for Indian plates
plate_cascade = cv2.CascadeClassifier("indian_license_plate.xml")

# Strict pattern: XX-00-XX-0000
strict_plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$')

def preprocess_image(roi):
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Use bilateral filter to preserve edges
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Increase contrast
    contrast = cv2.convertScaleAbs(blur, alpha=1.5, beta=0)

    # Adaptive thresholding (better for yellow/dark plates)
    thresh = cv2.adaptiveThreshold(contrast, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def clean_plate_text(text):
    # Remove non-alphanumeric and normalize spacing
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    # Validate against strict pattern
    if strict_plate_pattern.match(cleaned):
        return cleaned
    return None

def generate_frames():
    global recognized_plates
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, 1.1, 6)

        for (x, y, w, h) in plates:
            roi_color = frame[y:y + h, x:x + w]
            processed = preprocess_image(roi_color)

            text = pytesseract.image_to_string(processed, config='--psm 8')
            plate_number = clean_plate_text(text)

            if plate_number and plate_number not in recognized_plates:
                recognized_plates.append(plate_number)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            if plate_number:
                cv2.putText(frame, plate_number, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plates')
def plates():
    return jsonify(recognized_plates)

if __name__ == "__main__":
    app.run(debug=True)
