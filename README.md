# Face-Mask-Detection
This repository consist of an approach to reduce risk of Coronavirus spread. Wearing a mask is among the non-pharmaceutical intervention measures that can be used to cut the primary source of SARS-CoV2 droplets expelled by an infected individual. The outstanding performance of the proposed model is highly suitable for video surveillance devices.. Though should improve more on accuracy of the model in the future.

Dataset link : https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset

Training images : 4319                  
Test images     : 1698                       
Images resized to 300x300                                  

Weight file : checkpoint_ssd300.pth.tar which is trained and stored in a drive for further detection process

The VGG16 model is used as a base network to classify the objects and to turn the convolution neural network for detection process as regression problem ,SSD - Single Shot Detector model is used and stacked above for further predictions like bounding box position and confidence scores
It is used for the detection of objects in an image. Using a basic architecture of the VGG-16 architecture, the SSD can outperform other object detectors such as YOLO and Faster R-CNN in terms of speed and accuracy.

Finding ways to improve model performance further in the future!!

