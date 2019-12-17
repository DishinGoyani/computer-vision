# Facial Keypoint Detection
### Project Overview
This project is part of Udacity Computer vision nanodegree. In this project, you’ll build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face!  

Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition.  

<img src='images/key_pts_example.png' width=50% height=50% /> <img src='images/landmarks_numbered.jpg' width=30% height=30%/>

The project will be broken up into a few main parts in four Python notebooks:  

**Notebook 1** : Loading and Visualizing the Facial Keypoint Data  
 - Dataset: https://www.cs.tau.ac.il/~wolf/ytfaces/  
 - Custom Datasets, DataLoaders and Transforms  
 
**Notebook 2** : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints  
- CNN Architecture
    ```python
    Net(
      (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
      (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc1): Linear(in_features=57600, out_features=1000, bias=True)
      (fc2): Linear(in_features=1000, out_features=136, bias=True)
      (dropout): Dropout(p=0.2)
      (batch_norm): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    ```  
- Transform the dataset  
- Batching and loading data  
- Training  

**Notebook 3** : Facial Keypoint Detection Using Haar Cascades and your Trained CNN  
- Detect all the faces in an image using a face detector (Haar Cascade detector).  
- Pre-process those face images so that they are grayscale, and transformed to a Tensor of the input size that your net expects.  
- Use trained model to detect facial keypoints on the image.  

**Notebook 4** : Fun Filters and Keypoint Uses  

You can find these notebooks in the Udacity workspace that appears in the concept titled Project: Facial Keypoint Detection.   !
