🧠 Digit Recognizer - From Scratch with Numpy
This project implements a basic digit recognition system using only NumPy, trained on the popular Digit Recognizer dataset (similar to MNIST). The goal is to build a neural network from scratch — no deep learning libraries like TensorFlow or PyTorch — to truly understand the mathematics and mechanics behind neural networks.

This project demonstrates how to build and train a neural network **from scratch using only NumPy**. The model is trained on the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/), achieving high accuracy with a simple architecture and no external ML libraries.


********** DO : from utils import one_hot_encode, relu, relu_derivative, softmax, get_predictions, get_accuracy, initialize_parameters **********

---

## 📌 Features

- Implements forward & backward propagation manually
- Fully vectorized operations using NumPy
- Trains a 2-layer neural network
- Visualizes accuracy & loss over time
- Predicts on new digit images

---

## 🧠 Model Architecture

- **Input Layer:** 784 nodes (28x28 image)
- **Hidden Layer:** 64 neurons with ReLU activation
- **Output Layer:** 10 neurons (digits 0–9), Softmax

---

## 🚀 Training Details

| Detail         | Value               |
|----------------|---------------------|
| Optimizer      | Gradient Descent     |
| Loss Function  | Cross-Entropy        |
| Accuracy       | ~XX% on test data    |
| Epochs         | 500                  |
| Learning Rate  | 0.1                  |

## 📷 Prediction on Custom Images

You can also test the model on your own handwritten digits!

```python
from PIL import Image
img = Image.open("example_digits/3.png")
preprocessed = preprocess(img)
predict(preprocessed)

## Heyy its Parthiv , here is a note on all the mathematics used : -
This neural network is based on a standard two-layer feedforward architecture. The first step in the process is to take the input data and perform a linear transformation using weights and biases. This is followed by an activation function called ReLU (Rectified Linear Unit), which introduces non-linearity. The result is the output of the hidden layer.

The hidden layer’s output is then passed into another linear transformation followed by the softmax function. The softmax function converts the raw outputs into a probability distribution across the possible classes, allowing us to determine the model's prediction.

To measure how well the model is performing, we use a loss function known as cross-entropy. This function calculates the difference between the predicted probabilities and the actual labels.

Training the model involves a process called backpropagation. In this step, the model calculates how much each parameter (like weights and biases) contributed to the error. It then updates these parameters using an optimization technique called gradient descent, which adjusts the values slightly in the direction that reduces the error. This process is repeated over many iterations until the model becomes accurate in its predictions.

