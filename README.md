# Dog Breed Identification using Transfer Learning

Haoming Jin

## Project Overview

This project is a supplementary to my ["WeRateDogs Data Wrangling" project](https://github.com/carterjin/Twitter-WeRateDogs-Data-Wrangling), in which we used some results provided by Udacity which take dog pictures and predicts its dog breeds. Now I would like to implement this myself. I will build a pipeline that can be used within a web or mobile app for real-world images. Given an image, my algorithm will identify an estimation of the dog's breed. If given an image of a human, it will estimate the most resembling dog breed.

This Project also serves as my Capstone Project for Udacity Data Scientist Nanodegree. The training/validation/test data and bottleneck features data has already been given by Udacity, but I actually did it from scratch and want to show you how to do that. For example if you have only png files and labels, how would you do this.

The data is downloaded from [Dog Breed Identification Kaggle Competition](https://www.kaggle.com/c/dog-breed-identification/data). The original data is from [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). The data contains 10222 dog photos and labels indicating the breed, a total 120 different breeds.

Alternatively, in order to have this notebook run in Udacity, I am also including the data and methods I used for the Udacity Project: [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip), and differentiate which one to run by a variable: data_flag.

The data for human recognition is also provided by Udacity: [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

## Depedencies:

- keras 2.1.6
- tensorflow 1.14.0
- sklearn 0.19.1
- numpy
- pandas
- matplotlib
- openCV
- PIL


## Summary of Analysis

1. Read Dog Dataset and Preprocessing

We download the data and preprocessed it to make it usable for the CNN models.

2. Create a CNN to Classify Dog Breeds from Scratch

We built a CNN from scratch, the CNN consist of 3 hidden layers and subsequent maxpooling layers, and a final Dense layer. The test accuracy for the trained model is 1.9%. It is not very accurate but better than random guess.

3. Create a CNN to Classify Dog Breeds (using Transfer Learning)

We try two methods to make use of transfer learning. The first one is to use pre-trained model a feature extractor in model. The second one is extract bottleneck features first and use them to train the final layers. Either method the best model within VGG16, ResNet50, InceptionV3 is ResNet50, reaching 81.5% test accuracy.

4. Detect Humans and Dogs

For human detection, we use OpenCV's implementation of Haar feature-based cascade classifier to detect human faces in images. 99% of human files are detected as human, and 11% of dog files are mistakenly detected as human.

For dog detection, we again use ResNet50 model and imagenet pretrained weight to determine if predicted category is within all the dog categories. 100% of dog files are detected as dog, and 4% of human files are mistakenly detected as dogs.

5. Wrap up and the Final Algorithm

We pass an image sequentially though the dog detector and the human detector, and determine if the image is a dog, a human or neither. Then,

- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

## Files in this repo

### [Dog_Breed_Prediction.ipynb](https://github.com/carterjin/Dog-Breed-Identification-using-Transfer-Learning/blob/master/Dog_Breed_Prediction.ipynb)
This notebook provides the full analysis process and functional code for dog breed identification.

### [dog_breed_predict.py](https://github.com/carterjin/Dog-Breed-Identification-using-Transfer-Learning/blob/master/dog_breed_predict.py)
Using this python model that I have trained you can import this python file see prediction results on your pictures.

__How to use__:

1. Download the [model save file] (https://github.com/carterjin/Dog-Breed-Identification-using-Transfer-Learning/blob/master/resnet50_dog_predict_model) and extract them to the same directory as the python file.
2. Download the [Haars Cascade face detection pretrained weight](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml) and put it in a folder ```haarcascades``` under the same directory as the python file.
3. Import and run as following:
```
from dog_breed_predict import DogBreedPredict

pred = DogBreedPredict()
pred.predict_breed('Your image file path')
```
You should be able to get a result like this:

![](result_sample.png)

## Webapp

The webapp implementation is here:

[https://github.com/carterjin/dog-breed-app](https://github.com/carterjin/dog-breed-app)

you can follow the installation instructions and try it out yourself!

## References

1. Kaggle competition: https://www.kaggle.com/c/dog-breed-identification/data
2. Transfer Learning: https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
3. Cascade Classifier: https://docs.opencv.org/trunk/db/d28/tutorial_cascade_classifier.html
4. Human Detection: https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6
5. Imagenet index to labels: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
