# MNIST Handwritten Digit Classification using Neural Networks

## Project Overview

This project builds a simple neural network using Keras and TensorFlow to classify handwritten digits from the MNIST dataset. The MNIST dataset contains 60,000 training images and 10,000 test images, each representing one of the digits from 0 to 9. The model is a multi-class classifier that predicts the digit in the input image with high accuracy.

## Dataset

The dataset used in this project is the **MNIST dataset**, which consists of 28x28 grayscale images of handwritten digits (0-9). The dataset is split into:
- **Training set**: 60,000 images
- **Test set**: 10,000 images

The pixel values are normalized to fall between 0 and 1 for better training performance. The labels are one-hot encoded to be used in multi-class classification.

## Project Structure

The project involves the following steps:

1. **Data Loading and Preprocessing**
   - Load the MNIST dataset using `keras.datasets.mnist.load_data()`.
   - Normalize the image pixel values to a range of [0, 1].
   - One-hot encode the labels for multi-class classification.

2. **Model Architecture**
   - Build a neural network using Keras' Sequential API:
     - **Flatten Layer**: Converts 28x28 images into a 1D vector.
     - **Dense Layer**: A fully connected layer with 128 neurons and ReLU activation.
     - **Dropout Layer**: Prevents overfitting by randomly setting 20% of the neurons to zero during training.
     - **Output Layer**: 10 neurons with softmax activation for classifying 10 digits.

3. **Model Compilation and Training**
   - Compile the model using the Adam optimizer, categorical cross-entropy loss, and accuracy as a metric.
   - Train the model with 5 epochs and a validation split of 20%.

4. **Model Evaluation**
   - Evaluate the model on the test data, achieving an accuracy of **97.5%**.

5. **Model Saving**
   - The trained model is saved as `mnist_model.h5` for future use.

6. **Prediction and Visualization**
   - The trained model is loaded, and a random test image is selected.
   - The image is passed through the model for prediction, and both the true and predicted labels are displayed alongside the image.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

You can install the dependencies using:
```bash
pip install tensorflow keras numpy matplotlib
```

## Code Example

Here is an example of how the neural network is built and trained:

```python
# Build the neural network model
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units=10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_split=0.2)
```

## Model Performance

The model achieves around **97.5% accuracy** on the test data, making it highly effective for handwritten digit recognition.

## Usage

After training the model, you can load the model to make predictions on new images as follows:

```python
# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

# Predict the digit for a new image
predictions = model.predict(input_image)
predicted_label = np.argmax(predictions)
```

## Future Work

- **Model Improvements**: Experiment with deeper networks or other architectures like CNNs.
- **Hyperparameter Tuning**: Adjust learning rates, batch sizes, or number of epochs.
- **Deployment**: Deploy the trained model using frameworks like Flask, FastAPI, or TensorFlow Serving.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
