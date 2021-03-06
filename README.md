# Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder

Performed Anomaly Detection using a deep Learning auto-encoder (using h2o). 

Trained the auto-encoder (2 hidden layer x 400 neurons, Tanh activated) on a bunch of faces, and tested it out on an equal mix of faces and non-faces. During prediction, a non-face is an image that will have a high error of reconstruction (since the auto-encoder is trained to reconstruct faces.)

#####Confusion matrix:

![Confusion Matrix](https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/cm.png?raw=true)

An accuracy of a little over 80%.

**Modeled in R**

###Examples:

The top 5 predicted non-faces:

![nf1](https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/examples/nf1.png?raw=true)

![nf2](https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/examples/nf2.png?raw=true)

![nf3](https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/examples/nf3.png?raw=true)

![nf4](https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/examples/nf4.png?raw=true)

![nf5](https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/examples/nf5.png?raw=true)

The top 5 predicted faces:

![f1](https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/examples/f1.png?raw=true)

![f2](https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/examples/f2.png?raw=true)

![f3](https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/examples/f3.png?raw=true)

![f4](https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/examples/f4.png?raw=true)

![f5](https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/examples/f5.png?raw=true)




