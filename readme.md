## 학습 프레임 워크 구축 

### 1. 학습 데이터 다운로드 (구글드라이브 링크 클릭 후 YOLO_TRAIN_DATA.zip 다운로드)
*참고: 최종제출데이터셋.zip과 같은 내용이나 학습 구현을 용이하게 하기위해 재구성한 파일임
```
학습 데이터 구성

Train       |---- images
            |---- images_IR
            |---- labels

Validataion |---- images
            |---- images_IR
            |---- labels

Test        |---- images
            |---- images_IR
            |---- labels
```

### 2. 의존성 패키지 설치
```
conda create -n yolo_etri python=3.8
conda activate yolo_etri
(yolo_etri) pip install -r requirements.txt
(yolo_etri) conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

```

### 3. 학습 데이터 경로 지정

학습 데이터 폴더를 다음 코드를 이용하여 아래와 같이 구성
```
(yolo_etri) mkdir train_data 
(yolo_etri) mv train_data
(yolo_etri) unzip YOLO_TRAIN_DATA.zip
```

최종 폴더 구성은 다음과 같아야 함
```
data        |---- coco.yaml
            |---- myCustomData.yaml
            |---- ...

train_data  |---- Train --------|---- images
                                |---- images_IR
                                |---- labels

            |---- Validataion --|---- images
                                |---- images_IR
                                |---- labels
            
            |---- Test ---------|---- images
                                |---- images_IR
                                |---- labels
...

train_yolov5s_fusion.py
...


```
### 4. data/myCustomData.yaml 내용을 다음과 같이 수정
```
path: train_data
train: Train/images/
val: Validation/images/
test: Test/images/

# Classes
nc: 1  # number of classes
names: ['person']  # class names
```

### 5. 학습 시작
```
(yolo_etri) python3 train_yolov5s_fusion.py
```


