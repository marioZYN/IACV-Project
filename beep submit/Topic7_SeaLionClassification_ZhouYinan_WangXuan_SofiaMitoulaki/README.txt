IACV Project 2017/2018

Project Name: 7) Detection and Classifications of Sea Lions in Aerials Images
Supervisor: G. Boracchi - giacomo.boracchi@polimi.it

Team Member: Zhou Yinan 10551242
	     Sofia Mitoulaki 10597553
	     Wang Xuan 10549075

The files contained in this folder is the source code for the project. This project is based on a Kaggle competition.(https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count). The whole dataset can be downloaded there. The data set is quite large, containing around 15G training files and 86G testing files. In our project, we only use the 15G training files. Due to the size of the data set, here in the source code file we only include 4 images for illustration purpose. The code also works if you include more images in the data folder.

Software Requirement:

Anconda enviroment is recommended. 
Python 3.6.3(>=3.6.5)
tensorflow 1.8.0
keras with tensorflow as backend 2.1.6
sklearn 0.19.1
cv2 3.3.1
numpy 1.14.3
pandas 0.20.3

Hardware Requirement:

The code is developed in MacOS 10.13.4 and Ubuntu 16.04.4 
We didn't tested the code in Windows but it should work if anconda python env is used.
In order to train the whole dataset, we need to use Nvidia GPU and Cuda. But here we only have 3 images, CPU is enough. Our code automatically detects if a GPU is available. If so, it will use GPU as preferred choice. 


Instructions:

if Pycharm IDE is used, you can directly import the whole file as a project. Or you can run the code in terminal.

1. cd to current dir
2. python prepare.py 		(data pre-processing, generating train valid test patches, it may take a while)
3. python -m models.SealionClassifyKeras 	(train our model)
4. python -m models.VGG16Keras 		(do transfer learning)



Github repo

In order to reduce the size and simplify the zip file, we only contain the core training and testing code in this file. Other codes, like data 
visualization, model comparision, error analysis etc.. can be found in the following link:

https://github.com/marioZYN/IACV-Project





Thanks.
