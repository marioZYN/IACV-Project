# Assignment 1

Before using CNN to classify the image, we design a simple classifier (SVM, decision tree, LR etc.). The goal of it is to do binary classification, determining whether a given patch contains sealions or not. Basically there are two purposes of doing this phase:
1. calculate the baseline to compare with CNN solution
2. act as pre-processing phase of CNN to increase the performance

The General pipeline is the following:

patches -> pre-processing classifier -> CNN

We will discard the patches wihout sealions and feed our CNN only with patches containing sealions. We do not want to miss-kill legal sealion patches, so we need to design our binary classifier to achieve high recall.

In the code, we use logistic regression to classify patches. We manually create three features: average R, G, B values. Each patch is of dimension 96\*96\*3, so the features are calculated using the average value of each channel. In theory, all the patches with high value in Blue channel is a background patch. By using logistic regression, we can modify the threshold to do the precision & recall trade-off. In order to make recall high, we need to decrease the threshold.

Conclusion

The simple logistic regression classifier using R, G, B values as features only achieves a slightly higher score than the baseline (classify all the patches as sealion patches).   
In order to improve the score, we can try the following two methods:
* use a different algorithm
* create more features 

In the end, we decide not to use this pre-processing phase and only use CNN for classification. 
