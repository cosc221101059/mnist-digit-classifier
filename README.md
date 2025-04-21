# MNIST Digit Classifier 🧠✍️

This project demonstrates my **first neural network** using TensorFlow, trained to recognize handwritten digits from the classic **MNIST dataset**. It’s a beginner-friendly implementation that showcases how a simple feedforward neural network can achieve high accuracy on image classification tasks.

---

## 📦 Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a large database of 28x28 grayscale handwritten digits from 0 to 9.

We use TensorFlow's built-in dataset loader:

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
## 🛠️ Preprocessing
The image data is normalized to bring pixel values from a range of [0, 255] to [0, 1] for better model performance:
```python
x_train, x_test = x_train / 255.0, x_test / 255.0
```
## 🧠 Model Architecture
The model is built using tf.keras.Sequential with the following layers:

Flatten layer to reshape 28x28 images into 784-element vectors

Dense layer with 128 units and ReLU activation

Output Dense layer with 10 units and softmax activation (for digits 0–9)
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

```
## ⚙️ Compilation and Training
The model is compiled and trained using:
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=11)
```
## 📊 Evaluation
The model's performance is evaluated on the test dataset:
```python
model.evaluate(x_test, y_test)
```

## 💾 Saving the Model
The trained model is saved in both HDF5 and native Keras format:
```python
 model.save('hand_class.h5')
model.save('hand_class.keras')
```

## 📷 Sample Visualization
Visualize a sample digit from the training set:
```python

import matplotlib.pyplot as plt

plt.imshow(x_train[505])
plt.show()

print("Label:", y_train[505])
```
## 📌 Notes
## This is my first neural network, so it's meant as a learning project.

## The model performs surprisingly well even with a simple architecture.

