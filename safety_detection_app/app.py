from flask import Flask, render_template, request, redirect, url_for, Response, send_from_directory
import os
from werkzeug.utils import secure_filename
from detection_utils import SafetyGearDetector
import cv2
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}

# Initialize detector
detector = SafetyGearDetector('yolov5/runs/train/exp13/weights/best.pt')

# For real-time detection
camera = None
stop_camera = False


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def generate_frames():
    global camera, stop_camera
    camera = cv2.VideoCapture(0)
    while not stop_camera:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame, _ = detector.detect(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()
    camera = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/realtime')
def realtime():
    global stop_camera
    stop_camera = False
    return render_template('results.html', mode='realtime')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_camera')
def stop_camera_feed():
    global stop_camera
    stop_camera = True
    return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process based on file type
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Image processing
            image = cv2.imread(filepath)
            processed_image, _ = detector.detect(image)
            processed_filename = f"processed_{filename}"
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, processed_image)
            return render_template('results.html',
                                   mode='image',
                                   original=filename,
                                   processed=processed_filename)
        else:
            # Video processing
            processed_filename = f"processed_{filename}"
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            detector.process_video(filepath, processed_filepath)
            return render_template('results.html',
                                   mode='video',
                                   original=filename,
                                   processed=processed_filename)

    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)