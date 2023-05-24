from io import BytesIO
import numpy
from flask import Flask, request, redirect, render_template
from PIL import Image
import base64
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

allowed_exts = {'jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'}
app = Flask(__name__)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_DUPLEX


def check_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        if file and check_allowed_file(file.filename):
            img = Image.open(file.stream)
            image_processed = process_image(numpy.asarray(img))
            image_new = Image.fromarray(image_processed)
            with BytesIO() as buf:
                image_new.save(buf, 'jpeg')
                image_bytes = buf.getvalue()
            encoded_string = base64.b64encode(image_bytes).decode()
        return render_template('index.html', img_data="data:image/jpeg;base64," + encoded_string), 200
    else:
        return render_template('index.html', img_data=""), 200


def detect_emotion(img):
    emotions_list = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    print(loaded_model.predict(img))
    return emotions_list[np.argmax(loaded_model.predict(img))]


def process_image(img):
    global loaded_model
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights("model_weights.h5")
        loaded_model.make_predict_function()

    img = cv2.resize(img, (300, 300))
    gray_fr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_fr, 1.3, 5)

    for (x, y, w, h) in faces:
        fc = gray_fr[y:y + h, x:x + w]

        roi = cv2.resize(fc, (48, 48))
        determined_emotion = detect_emotion(roi[np.newaxis, :, :, np.newaxis])
        print(determined_emotion)

        cv2.putText(img, determined_emotion, (x, y-10), font, 1, (212, 6, 6), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0')
