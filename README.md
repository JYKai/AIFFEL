# Project : Drowsiness Detection & Emotion Classification
___
### 📣 개발 목적
- 교통사고의 대부분인 졸음운전을 예방
- 현재 자율 주행 자동차의 단계에서 는 아직 운전자의 주의가 필요한 단계이므로 운전자의 얼굴을 통한 감정 분석과 움직임 분석을 통해서 졸음운전을 사전에 예방

### 📣 기대 효과
- 졸음 운전을 사전에 예방할 수 있는 시스템 구축은 교통사고 유발 가능성을 낮추기에 사회적으로 좋은 영향을 기대할 수 있다.

### 📆 프로젝트 기간
- 2021.10.13 ~ 21.12.15

### 🛠 Tools
---
- Language : Python
- IDE : VScode / Jupyter Notebook
- Detecting Tool : OpenCV / Dlib
- Library : Tensorflow / Keras


### Colleagues
---
- [JYKcode](https://github.com/JYKcode)
- [RestHope](https://github.com/RestHope)
- [aacara](https://github.com/aacara)


### Datasets
---
- [Fer2013](https://www.kaggle.com/msambare/fer2013)
- [KETI 감정분류용 한국인 안면 dataset](https://aihub.or.kr/opendata/keti-data/recognition-visual/KETI-01-001)


### Process
---
![image](https://user-images.githubusercontent.com/88880041/145985351-a6f01f9e-65f3-4762-b98f-e0df66597040.png)

- 감정 분석, 졸음 판단, 경고 3가지의 과정을 통합하여 전체적인 분석 모델을 개발하는 방향으로 개발을 진행하였다.
- VGG 모델을 사용하였으며, 실시간 분석이라는 측면에서는 처리시간이 중요하기 때문에 모델의 구조를 단순화 시켰다.

### Structure
---
![2](https://user-images.githubusercontent.com/88880041/145986325-e8366773-5aa9-4ca2-955a-8f7c2352f900.png)

**Step 1. Emotion Classification**

 - cv2.VideoCapture()를 통해 매 Frame을 받아 온 뒤 Haarcascades 라이브러리를 통해 안면 인식을 한다. 
 - 안면 인식 후 모델에 적합한 크기로 만들어주기 위해 Resize를 해준 뒤에 predict를 하기 위해 Reshape을 해준다. 
 - 주어진 모델을 통해 Angry, Happy, Neutral 상태를 구분해준다.
 
**Step 2. Drowsy Detection**

 - cv2.VideoCapture()를 통해 매 Frame을 받아 온 뒤 dlib 라이브러리를 사용하여 얼굴 안면 인식과 얼굴의 랜드마크를 찾아 준다.
 - 얼굴의 랜드마크를 결정한 다음 얼굴의 랜드마크(x, y) 좌표를 NumPy 배열로 변환한다.
 - 왼쪽과 오른쪽 눈의 좌표를 추출하고 좌표를 사용하여 양쪽 눈의 눈 가로 세로 비율을 계산한다.
 - cv2.convexHull()과 cv2.drawContours()를 통해 왼쪽과 오른쪽 눈의 볼록한 부분을 계산하고 각각의 눈을 시각화한다.
 - 눈 가로 세로 비율을 임의로 정한 임계값 보다 낮은지 확인하고, 낮으면 졸음 판별 카운터를 늘린다. 
 
**Step 3. Warning**
- pygame 라이브러리의 mixer을 이용하여 해당 조건이 충족되면 주어진 음악파일을 재생시킵니다.

### Source Code
---
- run.py : 카메라를 통한 실시간 분석
- drowsy_landmark.py : 졸음 측정 시 기준이 되는 landmark 좌표 값을 설정
- crop_images.ipynb : 주어진 이미지들을 crop
- improve_images.ipynb : Histogram normalization, Blurring을 통한 이미지 성능 향상

### Development
---
- Image : Histogram normalization, Blurring이 아닌 Image sharpning 등의 다른 기법을 적용시켜 Dataset 자체의 성능을 향상
- Model : 실시간에 적용시킬 수 있는 다른 모델들을 사용해보지 못했다. 현재 실시간으로 사용될 만큼의 FPS가 나오지 않기 때문에 새로운 모델의 적용을 통해 지속적으로 FPS를 향상
- Effects : 졸음과 감정분석을 통한 추천시스템 구축과 행동 분석을 통한 복합적인 분석









