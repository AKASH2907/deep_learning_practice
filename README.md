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

## Dataset Description
1) MNIST - 70k images total with Train/Val split - 60k/10k - 10 classes
2) CIFAR10 - 60k images with Train/Test split - 50k/10k - 10 classes

## Codes
* [split_data.py](https://github.com/AKASH2907/deep-learning-with-keras/blob/master/split_data.py) - Generating X_train, Y_train, X_test, Y_test from new training data
* [alexnet.py](https://github.com/AKASH2907/deep-learning-with-keras/blob/master/imagenet%20models/alexnet.py) - AlexNet implementation from original Paper (used BatchNormalization instead of Local response normalization)
* [vgg16.py](https://github.com/AKASH2907/deep-learning-with-keras/blob/master/imagenet%20models/vgg16.py) - VGG 16/19 finetuning
* [inception_v3.py](https://github.com/AKASH2907/deep-learning-with-keras/blob/master/imagenet%20models/inception_v3.py) - Inception V3 finetuning network
* [inception_resnet_v2.py](https://github.com/AKASH2907/deep-learning-with-keras/blob/master/imagenet%20models/inception_resnet_v2.py) - Inception ResNet V2 finetuning network
* [c3d_features_generation.py](https://github.com/AKASH2907/deep-learning-with-keras/blob/master/feature_generation_videos/c3d_features_generation.py) - Video Level feature generation using Convolutional 3D model.
* [inception_v3_fetaures_generation.py](https://github.com/AKASH2907/deep-learning-with-keras/blob/master/feature_generation_videos/inception_pool3_feature_generation.py) - Frame Level feature generation using Inception V3 model.

## References
[1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, "[
Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)" arXiv preprint arXiv:1512.00567.

[2] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi, "[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)" arXiv preprint arXiv:1602.07261.

[3] Karen Simonyan, Andrew Zisserman, "[Very Deep Convolutional Networks for Large-Scale Image Recognition
](https://arxiv.org/abs/1409.1556)" arXiv preprint arXiv:1409.1556.

[4] Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, and Sudheendra Vijayanarasimhan, “[Youtube-8m: A large-scale video classification benchmark.](https://arxiv.org/abs/1609.08675)” arXiv preprint arXiv:1609.08675, 2016.

[5] Tran Du, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. "[Learning spatiotemporal features with 3d convolutional networks.](https://arxiv.org/abs/1412.0767)” In Proceedings of the IEEE International Conference on Computer Vision, pp. 4489-4497. 2015.

**Updating regularly**

