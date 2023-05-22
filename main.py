import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def detect_emotion(img):
    emotions_list = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    return emotions_list[np.argmax(loaded_model.predict(img))]


def main():
    img = cv2.imread("images/h13.jpg")
    cv2.imshow('Determined emotions', img)

    global loaded_model
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        loaded_model.load_weights("model_weights.h5")
        loaded_model.make_predict_function()

    gray_fr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_fr, 1.3, 5)

    for (x, y, w, h) in faces:
        fc = gray_fr[y:y + h, x:x + w]

        roi = cv2.resize(fc, (48, 48))
        determined_emotion = detect_emotion(roi[np.newaxis, :, :, np.newaxis])
        print(determined_emotion)
        cv2.putText(img, determined_emotion, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Determined emotions', img)
    cv2.waitKey(0)


#
# if __name__ == '__main__':
#     # SAMPLE_IMAGE_PATH = sys.argv[1]
#     img = cv2.imread(cv2.samples.findFile("images/a1.jpg"))
#     main(img)
if __name__ == '__main__':
    main()
