# Handwritten-Digit-Recognition

I have used mnist subset (images of handwritten digits from 0 to 9). The dataset is stored in a JSON-formated file mnist_subset.json. I developed the following supervised learning algorithms in Python, using only NumPy and SciPy packages:

• Logistic Regression model for binarized MNIST dataset using gradient descent to classify hand-written digits with an accuracy of 0.83 (in Logistic.py)
	•Run logistic_binary.sh
	•Output will be in logistic_binary.out

• Multi-class classification using one v/s rest classification and multinomial logistic regression function on MNIST dataset and achieved accuracy of 0.83 and 0.89 respectively (in Logistic.py)
	• Run logistic_multiclass.sh
	• Output will be in logistic_multiclass.out
