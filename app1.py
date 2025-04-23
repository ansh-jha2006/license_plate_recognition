from flask import Flask, render_template, Response, jsonify
import cv2
import pytesseract
import re
import numpy as np

app = Flask(__name__)
recognized_plates = []

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

plate_cascade = cv2.CascadeClassifier("indian_license_plate.xml")
strict_plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$')

def enhance_image(img):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization
    equalized = cv2.equalizeHist(gray)

    # Bilateral filter to reduce noise and keep edges
    blur = cv2.bilateralFilter(equalized, 11, 17, 17)

    # Sharpening
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(blur, -1, kernel)

    # Adaptive threshold for poor lighting or yellow plates
    thresh = cv2.adaptiveThreshold(sharpened, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def correct_perspective(roi):
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
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
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
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

def clean_plate_text(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
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

            # Correct tilted plates
            warped_roi = correct_perspective(roi_color)

            # Enhance for OCR
            processed = enhance_image(warped_roi)

            # Use improved Tesseract config
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 7')
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
