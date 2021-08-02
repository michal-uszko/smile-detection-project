# Smile detection
## General Info
The goal of this project was to make an application, which checks whether a person is smiling or not based on video stream from webcam. For this task, binary classification
was used. The neural network used for image classification utilizes ResNet50 architecture. Haar cascade for frontal face is used for face detection.

## Dataset
The neural network was using CelebA dataset for training, specifically align and cropped images and ```list_attr_celeba.csv``` file.
Official site from which you can download the dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Installation
Following libraries was used for this project:
* OpenCV (contrib version) (installation: https://pypi.org/project/opencv-python/)
* Tensorflow 2.x (installation: https://www.tensorflow.org/install)
* Keras (included with Tensorflow)
* Pandas (installation: https://pandas.pydata.org/docs/getting_started/install.html)
* imutils (installation: pip install imutils)
* splitfolders (installation: https://pypi.org/project/split-folders/)

## Setup
First of all, you will need to download CelebA dataset.
After that, run the command below (make sure, all file paths are correct):
```
python class_split.py
```
Then, split data into training and test sets:
```
python train_test_split.py
```
When the dataset is ready, you can start training the model (you might need to create an empty 'model' folder before running script below):
```
python train.py
```
Finally, you can check the results (you will need to connect a webcam to do so):
```
python predict.py
```
