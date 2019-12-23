# Image Captioning
### Project Overview
A neural network architecture to automatically generate captions from images. 
Network is trained on the Microsoft Common Objects in Context ([MS COCO](http://cocodataset.org/#home)) dataset, 
and tested network on novel images!  

![Image Captioning](/Image%20Captioning/images/image.png)  
<sub><sup>image: [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf) </sup></sub>

The project is structured as a series of Jupyter notebooks:  
- 0_Dataset.ipynb
- 1_Preliminaries.ipynb
- 2_Training.ipynb
- 3_Inference.ipynb  

## LSTM Decoder
In the project, we pass all our inputs as a sequence to an LSTM. A sequence looks like this: 
first a feature vector that is extracted from an input image, then a start word, then the next word, 
the next word, and so on!

> A completely trained model is expected to take between 5-12 hours to train well on a GPU; 
it is suggested that you look at early patterns in loss (what happens in the first hour or so of training) 
as you make changes to your model, so that you only have to spend this large amount of time training your 
final model.

You can find these notebooks in the Udacity workspace that appears in the concept titled Project: Image Captioning. 
