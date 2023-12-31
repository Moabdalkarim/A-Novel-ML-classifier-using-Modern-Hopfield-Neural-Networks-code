# A Novel ML classifier using Modern Hopfield Neural Networks

Mohamed Abdelkarim<sup>1</sup>, Hanan Ahmed Kamal<sup>2</sup>, Doaa Shawky<sup>2</sup>

<sup>1</sup> Siemens Digital Industries Software, Cairo, Egypt
<sup>2</sup> Faculty of Engineering, Cairo University, Cairo, Egypt

---

State-of-the-art classification neural networks, used for images and various data types, are complex and require significant energy and
computational resources relying on supervised gradient back-propagation. In contrast, the Hopfield Neural Network (HNN) is simpler, being
a single-layer, fully connected network that mimics the human brain's associative memory network, making it easy to implement and computationally
efficient. Its compatibility with oscillatory neural networks (ONNs) makes it ideal for light weigh machine learning applications in the Internet
of Things (IoT) era. Normally, HNN has been primarily used for associative memory aiding in image processing, pattern recognition, and more, but
this paper introduces it as a classifier. The proposed HNN classifier is adaptable to various datasets, including images and tabular, and requires
zero training time, making it suitable for resource-limited environments. It represents a significant leap in classification, with the highest accuracy
reported for HNN classifiers to the best of our knowledge, achieving 96\% accuracy on the MNIST dataset, a 36\% percentage improvement over previous models.


The full paper is available at:

## Hopfield_Classifier

## Requirements

The software was developed and tested on google colab.


As the development environment, [Python](https://www.python.org) 3.10.12 in was used.

## Installation

The recommended way to install the software is to use `pip/pip3`:

```bash
$ pip3 install git+https://github.com/ml-jku/hopular
```


## Installation

```python
import cupy as np
from hopfield_classifier import Hopfield_Classifier
```

## Usage

### Initialization

```python
# Initialize the Hopfield Classifier
# w_compress: weight compression factor (optional)
# PCA: number of PCA components (optional)
hnn = Hopfield_Classifier(w_compress=0.5, PCA=2)
```

### Training

```python
# x_train: training data features
# y_train: training data labels
# silent: if True, suppresses print statements (optional)
hnn.fit(x_train, y_train, silent=True)
```

### Prediction

```python
# x_test: test data features
# patch_size: size of the patch for energy computation (optional)
# silent: if True, suppresses print statements (optional)
predictions = hnn.predict(x_test, patch_size=-1, silent=True)
```

### Parameters

- `w_compress` (float, optional): Weight compression factor. Defaults to 0 (no compression).
- `PCA` (int/float, optional): Number of PCA components/percetange variance to apply. Defaults to 0 (no PCA).

### Methods

- `fit(x_train, y_train, silent=True)`: Train the network using the provided training data.
- `predict(x_test, patch_size=-1, silent=True)`: Predict the class of input instances.
- `compute_energy_ext(input_state, patch_size=-1)`: Compute the energy of the network for an external input state.

### Attributes

- `weights`: The weights matrix of the network after training.
- `pca_model`: The PCA model used for dimensionality reduction, if applicable.
- `energies`: A list storing energies during the network's operation.

## Example

```python
import cupy as np
from hopfield_classifier import Hopfield_Classifier

# Sample data
x_train = np.array([...])  # Training data features
y_train = np.array([...])  # Training data labels
x_test = np.array([...])   # Test data features

# Create Hopfield Classifier instance
hnn = Hopfield_Classifier(w_compress=0.5, PCA=2)

# Train the network
hnn.fit(x_train, y_train, silent=False)

# Predict using the network
predictions = hnn.predict(x_test, silent=False)

print(predictions)
```
- `Example google-colab notebook`: [google-colab notebook](Hopfield_Classifier.ipynb)
  
## License

[MIT](LICENSE.md)

