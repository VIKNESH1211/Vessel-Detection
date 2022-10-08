# Vessel-Detection 
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white) ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)


The project aims to train models which can detect marine vessels through images in ariel view and ideal view.

_______
## YoloV5_Method

### Data_set

The data set used for yolov5 method was the Ships-in-google-earth data set from kaggle, the annotations for the images in the data set are in XML format.

The data set was uploaded into roboflow for agumentation and preprocessing.
<p align="center">
  <img src="https://github.com/VIKNESH1211/Vessel-Detection/blob/main/YoloV5_method/Ship%20detect%20yolo%20-%20v1%202022-10-04%2011_30am%20-%20Google%20Chrome%2004-10-2022%2006_58_19%20PM.png?raw=true" width="700" alt="accessibility text">
</p>
The images in the data-set was resized to 640x640 , agumentation of counter-clockwise of 90deg and horizontal and vertical flip was done.

20% of the train data was used to make the validation data, therefore train-val seperation is done.

### Training

The model was trained using the yolov5s.pt weight which is available in the official ultralytics repository. The yolov5s.pt is the ideal weight which can be used for transfer learning on images with size of 640x640 using YoloV5.

Command used to train the model.
```sh
# To train the model
!python train.py --img 640 --batch 16 --epochs 100 --data '/content/drive/MyDrive/yolo v5 ship detection/custom.yaml' --weights yolov5s.pt --cache .
```
The model was succesfully trained and weights was obtained. <a href="https://github.com/VIKNESH1211/Vessel-Detection/blob/main/YoloV5_method/weight.pt" target="_blank">weight</a> 

**The graphs below show the training metrics.** The graphs was obtained using TensorBoard
<p align="center">
  <img src="https://github.com/VIKNESH1211/Vessel-Detection/blob/main/YoloV5_method/results.png?raw=true" width="700" alt="accessibility text">
</p>

### Detection

The model was tested on test images which were never before exposed to the model. The model returned a confidence score above 0.9 for every bounding box in the dectection.
Command used for Detection.
```sh
# To Detect.
!python detect.py --weights "/content/drive/MyDrive/yolo v5 ship detection/exp/weights/last.pt" --img 640 --source "/content/test"
```
<p align="center">
  <img src="https://github.com/VIKNESH1211/Vessel-Detection/blob/main/YoloV5_method/Detection/boat130.png" width="700" height="700" alt="accessibility text">
</p>
<p align="center">
  <img src="https://github.com/VIKNESH1211/Vessel-Detection/blob/main/YoloV5_method/Detection/boat138.png" width="700" alt="accessibility text">
</p>
<p align="center">
  <img src="https://github.com/VIKNESH1211/Vessel-Detection/blob/main/YoloV5_method/Detection/boat152.png" width="700" alt="accessibility text">
</p>
<p align="center">
  <img src="https://github.com/VIKNESH1211/Vessel-Detection/blob/main/YoloV5_method/Detection/boat179.png" width="700" alt="accessibility text">
</p>

___________
## Classifier

### Data_Set

The data set used to build the classifier is the 'Game of Deep Learning : Ship datasets' from kaggle https://www.kaggle.com/datasets/arpitjain007/game-of-deep-learning-ship-datasets.

The data set had 6000+ images with its labels in a csv format.

In the preprocessing , the images loaded into list and the images was resized/reshaped into (100 , 100, 3) and the list was the converted into an array , then by using sklearn's train_test_split twice the data was split into train test and val. And the labels in the csv were in the format of {'Cargo': 1,
'Military': 2,
'Carrier': 3,
'Cruise': 4,
'Tankers': 5} which was transformed into [0. 1. 0. 0. 0. 0.] , [0. 0. 1. 0. 0. 0.]  ,[0. 0. 0. 0. 1. 0.] ,etc using keras.utils.to_categorical.

### Training

The model was trained using the above preprocessed data and mobilenetv2 CNN architecture
```sh
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(6,activation='softmax')

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
```
The number of epoch used was 20 , the optimizer used is RMSprop with a learning rate of 0.00001 and the loss function used is Categoricalcrossentropy.
```sh
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
model.fit(X_train , Y_train , epochs=20 , validation_data=(X_val, Y_val), verbose = 1)            
```
The model achived an accuracy of 0.9824 in train data , an accuracy of 0.8144 in Validation data and an accuracy of 0.8131 in model.evaluate() using test data.
<div align="center" style="background-color:#FFFFFF;">
  <img src="https://github.com/VIKNESH1211/Vessel-Detection/blob/main/Classifier/accuracy.png" width="700" alt="accessibility text">
</div>
<p align="center" style="background-color:#FFFFFF;">
  <img src="https://github.com/VIKNESH1211/Vessel-Detection/blob/main/Classifier/loss.png" width="700" alt="accessibility text">
</p>
The training metrics were as above attached graph.
