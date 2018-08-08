# Handwritten-Digit-Recognition

This work was done as part of CSCI-567 (Machine Learning) coursework at USC.
I have used mnist subset (images of handwritten digits from 0 to 9). The dataset is stored in a JSON-formated file mnist_subset.json. I developed the following supervised learning algorithms in Python, using only NumPy and SciPy packages:

• Logistic Regression model for binarized MNIST dataset using gradient descent to classify hand-written digits with an accuracy of 0.83 (in Logistic.py) <br />
<b>Run</b> python3 logistic.py --type binary --output logistic_binary.out <br />
Output will be in logistic_binary.out
	
• Multi-class classification using one v/s rest classification and multinomial logistic regression function on MNIST dataset and achieved accuracy of 0.83 and 0.89 respectively (in Logistic.py) <br />
<b>Run</b> python3 logistic.py --type multiclass --output logistic_multiclass.out <br />
Output will be in logistic_multiclass.out

• Multi-Layer Perceptron (in dnn_mlp.py and dnn_misc.py) for 10-class classification problem, with various combinations of dropout rate. The network structure is input --> linear --> relu --> dropout --> linear --> softmax_cross_entropy loss <br />
<b>Run</b> python3 dnn_mlp.py --dropout_rate 0.5
    

• Convoluted Neural Network for 10-class classification problem, with various combinations of learning rate, momentum, weight decay and dropout rate (in dnn_cnn.py and dnn_misc.py). The network structure is input --> convolution --> relu --> max pooling --> flatten --> dropout --> linear --> softmax_cross_entropy loss
<br /><b>Run</b> python3 dnn_cnn.py --alpha 0.9 <br/><br/>
Further extended this to forma deeper architecture (CNN2 in dnn_cnn_2.py and dnn_misc.py). The network structure is input --> convolution --> relu --> max pooling --> convolution --> relu --> max pooling --> flatten --> dropout --> linear --> softmax_cross_entropy loss<br />
<b>Run</b> python3 dnn_cnn_2.py --alpha 0.9

• Adaboost and Logitboost, both with accuracies 0.95, on binarized MNIST data (in boosting.py and boosting_test.py) <br />
<b>Run</b> python3 boosting_test.py

• Linear support vector machine (SVM) using Pegasos algorithm for binary classification, giving an accuracy of 0.82  <br />
<b>Run</b> python3 pegasos.py

• Principal Component Analysis (PCA) for image compression (black and white images)
<b>Run</b> python3 pca.py
This will generate pca_output.txt


