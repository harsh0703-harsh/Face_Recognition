from flask import Flask, render_template, request
import requests
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('face_recog.h5')

## Croped Face Function
def face_extractor(img):
    
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return None
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


## Making Flask Application

app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        video_capture = cv2.VideoCapture(0)
        while True:
            _, frame = video_capture.read()
            face=face_extractor(frame)
            if type(face) is np.ndarray:
                face = cv2.resize(face, (224, 224))
                im = Image.fromarray(face, 'RGB')
                img_array = np.array(im)
                img_array = np.expand_dims(img_array, axis=0)
                pred = model.predict(img_array)
                name="Unknown"
                if(np.argmax(pred,axis=1)==0):
                    name='Harsh'
                    cv2.putText(frame,name, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (243,0,0), 2)
                else:
                    name='Jiya'
                    cv2.putText(frame,name, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (243,255,201), 2)
            else:
                cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
    return "Its Completed"

## code is ready, so run the application

if __name__=="__main__":
    app.run(debug=True)