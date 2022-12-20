# DIY FaceApp
Documentation of everything I've covered so far
## [Course 1 : Deep Learning and Neural Networks](https://github.com/mdoshi2612/DIY-FaceApp/tree/main/Checkpoints/Neural%20Networks%20and%20Deep%20Learning)
* Week 1 involved a basic overview of what [neural networks](https://www.ibm.com/cloud/learn/neural-networks) are and how deep learning works. 
* Week 2 involved an example where the outputs were binary i.e. the output could only take 2 values - A or B
* We then looked at logistic regression as a algorithm and uncovered the math that went into deep learning algorithms. This involved designing a cost function where then we used two dimensional gradient descent to reduce the cost function to a minimum value i.e training the model.
* Linear Regression only involved using the sigmoid function to find the activations of the next layer
* Week 3 involved more of the concepts of vectorization and how it would be easier to vectorize all training examples rather than looping through all the training set.
* This was achieved by stacking all the training examples as column vectors next to each other.
* We also looked at different activation functions and found out why they might or might not be more useful than the sigmoid activation function.
* Towards the end we saw the implementation of the concepts in a neural network with one hidden layer.
* Week 4 involved more practical work as we implemented the entire neural networks with a arbitrary number of layers.
* We also looked at the hyperparameters and parameters of the neural network which brings the introduction to the second course of this specialization
## [Course 2 : Hyperparameter Tuning, Regularization and Optimization](https://github.com/mdoshi2612/DIY-FaceApp/tree/main/Checkpoints/Hyperparameter%20Tuning%2C%20Regularization%20and%20Optimization)
* Course 2 involved hyperparamter tuning and how to effeciently lower the cost function quickly and optimization of the algorithms to speed up training
* Week 1 involved looking at the model and seeing how we can reduce overfitting during training.
* This mainly involved [L<sub>2</sub> regulariztion](https://papers.nips.cc/paper/1991/file/8eefcfdf5990e441f0fb6f3fad709e21-Paper.pdf) and using dropout and inverted dropout on the training samples to reduce overfitting
* Week 2 involved looking at different algorithms like dividing our entire training sample to different mini-batches to reduce the time it takes to update parameters
* We also looked at exponentially weighted averages and gradient descent with momentum to get to the global minima quicker.
* Finally we looked at [Adam's algorithm](https://arxiv.org/pdf/1412.6980.pdf) and RMS prop and implemented them in this week's assignment.
* Week 3 we looked  at batch normalization and looked at different deep learning frameworks like [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/all_symbols)
* For reading up more on optimizing deep learning algorithms [Click Here](https://ruder.io/optimizing-gradient-descent/) 
## Course 3 : Structuring Machine Learning Projects
* This course introduced me to various strategies to improve my deep learning model by making smarter decisions that enable us to improve the performance of our neural network.
* For example, let's say your network didn't have perform very well on the test set and you might have several ideas to improve your model's performance, like say collecting more data, training the algorithm longer, using Adam's algorithm instead of SGD, dropout, etc.
* What you may not realise is that some of these techniques might not greatly improve model performance and you might just end up wasting time.
* One of the ideas in DL is orthogonalisation. This specifies that making one change to a network will only change one of the metrics which makes it easier to test one problem at a time. 
* For example, if your algorithmn doesn't perform well on the training set, then you might consider making your network a bit deeper or using Adam's algorithm instead of using gradient descent.
* Another example is if your algorithm performs well on the training set but not as well on the dev set, you could try dropout or L<sub>2</sub> regularization.
* One useful idea adopted by ML engineers is setting up a single number evaluation metric; what this does is quickly allows you to select the most optimal algorithm out of 10-15 you might've tried. 
* Maybe to combine multiple metrics you could use satisficing and optimizing metrics. Let's say A takes 80 milliseconds, B takes 95 milliseconds, and C takes 1,500 milliseconds to classify an image. You want the model with the most accuracy, but you also want it to classify an image really fast.
* What you could you do in this case is have a criteria where you image must be classified within 100 ms and then pick the highest accuracy. Here time is sastificing metric while accuracy is a optimizing metric.
* One of the more interesting ideas is how to split your dev and test tests. Let's say you're working on a cat classifier with your data from 8 countries. You could pick your dev set in such a way that it contains data from 4 random countries while the test set contains rest of the data. 
* Turns out this is a bad way of dividing as now your test and dev sets are from different distributions. The better way would be to shuffle the data and divide into the dev and test set.
<<<<<<< HEAD
* Hello
=======
## Course 4 : Convolutional Neural Networks
* We've worked with images before, and we see flattening images of 64 by 64 pixels over 3 channels gives you 12288 neurons in each layer. This is a relatively small number of parameters but in case you work with high res images this number might blow up to billions of parameters over the entire model hence for proccessing images a new type of neural network had to be introduced i.e. CNNs.
* Hence a new operation known as convolutions was designed that uses filters to find the input to the next layer. To read up on the various operations in CNNs [click here](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
* Next we just look at the various types of classic networks that have been designed by researchers as case studies.
* [ResNets paper](https://arxiv.org/pdf/1512.03385.pdf); To get a better understanding watch this [video](https://www.coursera.org/learn/convolutional-neural-networks/lecture/HAhz9/resnets)
* One interesting usage of CNNs is the use of 1 by 1 convolutions and here lies the main idea of [Inception Networks](https://arxiv.org/pdf/1409.4842.pdf) and refer coursera for a better understanding as well.
* Using CNNs, we can also detect objects in an image. This can be done for autonomous driving cars where you might have to detect other vehicles, pedestrians or motorcycles. This feature can be implemented using a different vector label which specifies the probability of each object being there in an image and other variables b<sub>x</sub>, b<sub>y</sub>, b<sub>h</sub> and b<sub>w</sub> to make a box around that part of the image.
* To find the object that usually does not cover the entire image, the first idea we get is to take small crops of images from the main image and run a CNN over that image to find if the object exists or not. This is extremely expensive in terms of computations and hence we use a convolution to determine if the object exists in the image.
* What we do here is we divide the entire image into grids and assign the object to that grid which contains it's midpoint and then use [YOLO Algorithm](https://arxiv.org/pdf/1506.02640.pdf).
* For face detection, we encounter a different problem. Usually we only have one image of someone's face and hence the neural network needs to learn using one training example itself and hence this is called one-shot learning.
* To get around this problem, we instead develop sort of a similarity function that we can use to find the similarity between two faces.
* Hence for any input image x<sub>i</sub> we want a neural network that outputs an encoding f(x<sub>i</sub>). Hence if two faces are different we want the distance between their encodings to be as far apart as possible and as close as possible if it's the same person.
* To train this kind a neural network we use a special type of cost function called triplet [cost function](https://en.wikipedia.org/wiki/Triplet_loss) and to see a video on this cost function [click here](https://www.coursera.org/learn/convolutional-neural-networks/lecture/HuUtN/triplet-loss)
## Pytorch
* I created a very basic CNN using 2 convolutions and 3 fully connected layers to identify the CIFAR 10 dataset using pytorch.
* Link to the tutorial [here](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

>>>>>>> 6a37622d576c346dee34909fdec92db68361b4a0
