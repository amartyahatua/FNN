import torch
from torchvision import datasets, transforms
from keras.datasets import mnist
import numpy as np
from typing import TYPE_CHECKING, Callable, List, Dict, Optional, Tuple, Union


def to_categorical(labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def preprocess(
        x: np.ndarray,
        y: np.ndarray,
        nb_classes: int = 10,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances.
    :param y: Labels.
    :param nb_classes: Number of classes in dataset.
    :param clip_values: Original data range allowed value for features, either one respective scalar or one value per
           feature.
    :return: Rescaled values of `x`, `y`.
    """
    if clip_values is None:
        min_, max_ = np.amin(x), np.amax(x)
    else:
        min_, max_ = clip_values

    normalized_x = (x - min_) / (max_ - min_)
    categorical_y = to_categorical(y, nb_classes)

    return  categorical_y


def load_mnist_digit():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)

    x_train = dataset1.data.numpy()
    y_train = dataset1.targets.numpy()
    x_test = dataset2.data.numpy()
    y_test = dataset2.targets.numpy()

    min_, max_ = 0.0, 1.0
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    y_train = preprocess(x_train, y_train)
    y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_