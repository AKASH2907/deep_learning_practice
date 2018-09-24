# deep-learning-with-keras
keras implemetation of deep learning models

**If the Jupyter notebook file is not loading, please load the notebook link on [NBViewer](https://nbviewer.jupyter.org/). It is the problem from Github side.**

## MNIST Handwritten digits
Model/Accuracy| Training | Validation | Test
------------- | -------- | ---------  | ----------
Simple  |  89.77 | 90.67 | 90.59
Two Layers  |  93.03 |93.24  |93.28
Dropout | 88.41  |93.48 | 93.41
Adam  |   99.86 | 97.65 | 97.84

## Convolutional Neural Netwroks (CNNs)
1) MNIST - LeNet architecture - Accuracy - Train/Val/Test - 99.9/99.2/99.2
2) CIFAR10 - 

## Important Codes
* [split_data.py](https://github.com/AKASH2907/deep-learning-with-keras/blob/master/split_data.py) - Generating X_train, Y_train, X_test, Y_test from new training data
* [alexnet.py](https://github.com/AKASH2907/deep-learning-with-keras/blob/master/alexnet.py) - AlexNet implementation from original Paper (used BatchNormalization instead of Local response normalization)
* [vgg16.py](https://github.com/AKASH2907/deep-learning-with-keras/blob/master/vgg16.py) - VGG 16/19 implementation using keras.io.applications


## References
[1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, "[
Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)" arXiv preprint arXiv:1512.00567. <br />
[2] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi, "[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)" arXiv preprint arXiv:1602.07261. <br />
[3] Karen Simonyan, Andrew Zisserman, "[Very Deep Convolutional Networks for Large-Scale Image Recognition
](https://arxiv.org/abs/1409.1556)" arXiv preprint arXiv:1409.1556 <br/ >

**Updating regularly**

