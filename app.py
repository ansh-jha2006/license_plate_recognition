from flask import Flask, render_template, Response, jsonify
import cv2
import pytesseract
import re
import numpy as np
from difflib import get_close_matches  # For matching invalid state codes

app = Flask(__name__)
recognized_plates = []

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load custom cascade for Indian plates
plate_cascade = cv2.CascadeClassifier("indian_license_plate.xml")

# Strict pattern: XX-00-XX-0000
strict_plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$')

# List of valid Indian state codes
valid_state_codes = [
    "AP", "AR", "AS", "BR", "CH", "CT", "DL", "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML",
    "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
]

def correct_perspective(roi):
    # Detect edges using Canny
    edges = cv2.Canny(roi, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(roi, M, (maxWidth, maxHeight))
            return warped
    return roi

def validate_and_correct_state_code(plate_text):
    # Extract the first two letters (state code)
    state_code = plate_text[:2]
    if state_code in valid_state_codes:
        return plate_text  # Valid state code

    # Find the closest match for the state code
    closest_match = get_close_matches(state_code, valid_state_codes, n=1, cutoff=0.6)
    if closest_match:
        corrected_code = closest_match[0]
        return corrected_code + plate_text[2:]  # Replace the state code with the corrected one
    return None  # Invalid plate

def preprocess_image(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    contrast = cv2.convertScaleAbs(blur, alpha=1.5, beta=0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(contrast, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Apply morphological closing to connect text parts
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph

def clean_plate_text(text):
    # Remove all non-alphanumeric characters and uppercase
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())

    # Early return if length is off (Indian plates usually 10 characters)
    if len(cleaned) != 10:
        return None

    # Validate and correct state code
    corrected = validate_and_correct_state_code(cleaned)
    return corrected


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
            
            # Correct perspective for tilted plates
            roi_corrected = correct_perspective(roi_color)
            processed = preprocess_image(roi_corrected)

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