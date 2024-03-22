import numpy as np
import cv2
import tensorflow as tf
import pygame

from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time
import dlib
import threading
import imutils
from threading import Thread
from imutils import face_utils
from tensorflow.keras.preprocessing.image import img_to_array
from drowsy_landmark import eye_aspect_ratio
from collections import Counter

# model : lightvgg
model = Sequential()
model.add(Conv2D(16, (3, 3), padding="same", activation="relu", input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax', kernel_regularizer = regularizers.l2(0.01)))

# pygame을 통한 음악파일 작동
pygame.mixer.init()
pygame.mixer.music.load('fire-truck.wav')

# 운전자 상태 분석 수집 변수
status = 'Awake'
COUNTER_limit = 60
sign = None
color = None

# 프레임 카운터와 알람이 울리는지 여부를 나타내는 bool 초기화
COUNTER = 0

# 왼쪽, 오른쪽 눈 및 입의 랜드마크 index를 각각 잡는다.
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# drowsy face detector and the facial landmark predictor create
dro_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# emotion face detector create and load model, emotion labels
emo_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EMOTIONS = ['angry', 'happy', 'neutral']

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
outputFrame = None
outputProb = None
lock = threading.Lock()

cap = cv2.VideoCapture(0)
time.sleep(1.0)

# model.load_weights('lightvgg_model_e300_b256_d5_r_1212_t4.h5')

if not cap.isOpened():
    raise IOError("Cannot open webcam")

prevTime = 0

while True:

    ret, frame = cap.read()

    #현재 시간 가져오기 (초단위로 가져옴)
    curTime = time.time()

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # detect faces in the grayscale frame
    canvas = np.zeros((120, 450, 3), dtype='uint8')
    dro_rects = dro_detector(gray, 0)
    emo_rects = emo_detector.detectMultiScale(gray, scaleFactor=1.1,
                                                  minNeighbors=5, minSize=(30, 30),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
    # emotions detections
    if len(emo_rects) > 0:
        rect = sorted(emo_rects, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = rect

        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # preds
        preds = model.predict(roi)[0]   
        label = EMOTIONS[preds.argmax()]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            w = int(prob * 300)
            cv2.rectangle(canvas, (5, (i * 35) + 5),
                              (w, (i * 35) + 35), (0, 0, 225), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (255, 0, 0), 2)
    
    # loop over the drowsy face detections
    for dro in dro_rects:
        # 얼굴의 랜드마크를 결정한 다음 얼굴의 랜드마크(x, y) 좌표를 NumPy 배열로 변환한다.
        shape = predictor(gray, dro)
        shape = face_utils.shape_to_np(shape)   

        # 왼쪽과 오른쪽 눈의 좌표를 추출하고 좌표를 사용하여 양쪽 눈의 눈 가로 세로 비율을 계산한다.
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        eyes = (leftEye + rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # 왼쪽과 오른쪽 눈의 볼록한 부분을 계산하고 각각의 눈을 시각화한다.

        # cv2.convexHull()를 활용해 윤곽선에서 블록 껍질을 검출한다.
        # cv2.convexHull(윤곽선, 방향)을 의미한다.
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)     

        # cv2.drawContours()을 이용하여 검출된 윤곽선을 그린다.
        # cv2.drawContours(이미지, [윤곽선], 윤곽선 인덱스, (B, G, R), 두께, 선형 타입)을 의미한다.
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 눈 가로 세로 비율이 깜박임 임계값보다 낮은지 확인하고, 낮으면 깜박임 프레임 카운터를 늘립니다.
        if ear < 0.23:
            if label == 'happy' or label == 'angry':
                continue
            else:
                COUNTER += 1
                color = (0, 0, 255)
                status = 'drowsy'

        else:
            COUNTER = COUNTER - 1
            status = 'Awake'

            if(COUNTER < 0):
                COUNTER = 0

        sign = status + ', Sleep count : ' + str(COUNTER) + ' / ' + str(COUNTER_limit)

        if( COUNTER > COUNTER_limit ):
            frame = gray
            if (pygame.mixer.music.get_busy()==False):
                pygame.mixer.music.play()

        cv2.putText(frame, sign , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)

    print("Time {0} ".format(sec))
    print("Estimated fps {0} ".format(fps))

    str_fps = "FPS : %0.1f" % fps
    cv2.putText(frame, str_fps, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.imshow('Video', cv2.resize(frame,(1600,960), interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()