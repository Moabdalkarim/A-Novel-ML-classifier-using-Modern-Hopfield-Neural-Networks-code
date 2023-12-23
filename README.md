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

## Requirements

The software was developed and tested on google colab.


As the development environment, [Python](https://www.python.org) 3.10.12 in was used.

## Installation

The recommended way to install the software is to use `pip/pip3`:

```bash
$ pip3 install git+https://github.com/ml-jku/hopular
```

## Usage

Hopular has two modes of operation:

- `list` for displaying various information.
- `optim` for optimizing Hopular using specified hyperparameters.

More information regarding the operation modes is accessible via the `-h` flag (or, alternatively, by `--help`).

```bash
$ hopular -h
```

```bash
$ hopular <mode> -h
```

To display all available datasets, the `--datasets` flag has to be specified in the `list` mode.

```bash
$ hopular list --datasets 
```

Optimizing a Hopular model using the default hyperparameters is achieved by specifying the corresponding dataset in the
`optim` mode.

```bash
$ hopular optim --dataset <dataset_name>
```

## Examples

To optimize a Hopular model on the `GlassIdentificationDataset` using the default hyperparameters, only the dataset
name itself needs to be specified. More details on the default values are available in the
[console interface](hopular/interactive.py) implementation.

```bash
$ hopular optim --dataset "GlassIdentificationDataset"
```

Optimizing a smaller Hopular model on the `GlassIdentificationDataset` utilizing only `4` modern Hopfield networks, `2`
iterative refinement blocks, and a scaling factor of `10` is achieved by manually specifying the respective
hyperparameters.

```bash
$ hopular optim --dataset "GlassIdentificationDataset" --num_heads 4 --num_blocks 2 --scaling_factor 10
```

## Disclaimer

The [datasets](hopular/auxiliary/resources), which are part of this repository, are publicly available and may be
licensed differently. Hence, the [LICENSE](LICENSE) of this repository does not apply to them. More details on the
origin of the datasets are available in the accompanying paper.

## License

This repository is MIT-style licensed (see [LICENSE](LICENSE)), except where noted otherwise.
