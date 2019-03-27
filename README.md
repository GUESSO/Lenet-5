# Lenet-5
This is  lenet-5 network implemented without any Machine Learning framework by myself with python 2.7 according to the thesis http://219.238.82.130/cache/5/03/yann.lecun.com/b1a1c4acb57f1b447bfe36e103910875/lecun-01a.pdf, however I made some changes including: 
1. Use activation function ReLU instead of tanh; 
2. Every kernel in C3 layer in the paper is designed separately to convolute different feature maps, which is in order to get rid of symmetry of the layer and to save computational capacity due to the speed of computers in 1998.I take place of C3 with a traditional convolutional 3D layer .
3. A convolutional layer is conventionally set with a bias parameter, but according to some paper and blog, bias doesn't matter with the final result, so I remove the parameter bias.



## Start Method

1.Install "matplotlib","numpy" lib with pip.

2.Open the project with pycharm, otherwise you will need to install all the python package I implemented.

3.Open the file "testLenet.py" ,the function test_lenet5_class() is the inlet of all program.The class Lenet5__C

is has a default bool parameter "is_debug", if you set it as True, it will only train 100 data with batch size 10.

Else, it will train all the data in MNIST data set.

4.Whether "is_debug" is True or False, the program will draw a picture for every epoch dynamically according to the train result of every batch. After an epoch is over, the diagraph drawn will be saved in directory "rstImg" automatically. Also if the accuracy is >80%, the weights got by training will be saved in file,you can easily use the function "load_weights" to recur the result.



## Result display

I trained the network using three different optimizing algorithm SGD,Ada,Momentum (Adam is not used because it takes some time to implement in convolutional layer and I am little busy).The result is as following, while the red line represents the average loss, the blue line represents the average accuracy for a batch.

1.SGD result

training

![Image text](https://raw.githubusercontent.com/GUESSO/Lenet-5/master/train0_SGD.png)

testing

![Image text](https://raw.githubusercontent.com/GUESSO/Lenet-5/master/test_SGD.png)

2.Momentum

training

![Image text](https://raw.githubusercontent.com/GUESSO/Lenet-5/master/trainMom.png)

testing

![Image text](https://raw.githubusercontent.com/GUESSO/Lenet-5/master/testMom.png)

3.Ada

training

![Image text](https://raw.githubusercontent.com/GUESSO/Lenet-5/master/train_Ada.png)

testing

![Image text](https://raw.githubusercontent.com/GUESSO/Lenet-5/master/test_Ada.png)

As the pictures show above, I get ideal result using SGD,and  I'm pretty sure about why Ada is not eligible:

I observe the gradient in late batches, the gradient disappear due to the special algorithm of Ada.I'm wondering why momentum looks so inefficiently.If anyone has some idea you could send email to 18401605721@163.com.