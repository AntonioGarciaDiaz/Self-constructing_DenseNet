Self-constructing DenseNet with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Algorithm for automatically building `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`__ (DenseNets).

A modification of `Illarion Khlestov's TensorFlow implementation of DenseNets. <https://github.com/ikhlestov/vision_networks>`__

The algorithm that is currently implemented is based on the early stages of `the EMANN self-structuring algorithm by Salomé and Bersini (1994).
<https://ieeexplore.ieee.org/document/374473>`__
This simple algorithm is based on the evolution of two features, which are here called "connection strength" and "source connectivity".

The **connection strength** (CS) of a layer :math:`l` with a previous layer :math:`s` (between the previous block's output and :math:`l-1`)
is the mean of the absolute kernel weights associated with the connection :math:`s-l` (sum of the weights divided by num of weights).
Usually this value is normalised - that is, divided by the max CS for a set value of :math:`s` or :math:`l`, to give it a value between 0 and 1.

The **source connectivity** of a layer :math:`l` is also a value between 0 and 1. It expresses how many of the connections recieved by :math:`l` are treated as 'useful enough' by it.
For each connection from a previous layer :math:`s` (between the previous block's output and :math:`l-1`), the connectivity is increased by :math:`1/(l+1)` if the connection's CS is :math:`\geq 0.67` * the max CS for :math:`s`.

In the current self-constructing algorithm, a new layer is added to the last block every :code:`asc_thresh` epochs, until at least one layer in the block has a "source connectivity" equal to 1. Here :code:`asc_thresh` is a constant parameter with a default value of 10.
Experiments are currently being undertaken to implement a more efficient self-constructing algorithm.

**N.B.:** It is possible to specify a prebuilt initial structure on which the algorithm runs (different than one block with one layer). A parameter called :code:`layer_num_list` (or :code:`lnl`) is used for defining the prebuilt blocks with their number of layers.

Example: :code:`--layer_num_list '12,12,12'` would mean that three blocks with 12 layers are created (the default architecture in Illarion Khlestov's implementation), and the algorithm starts building inside the third block.

The default value for this parameter is :code:`'1'`. That is, the algorithm begins with only one block, containing a single layer, and starts adding new layers inside this block.

Two types of DenseNets are available:

- DenseNet - without bottleneck layers
- DenseNet-BC - with bottleneck layers

Each model can be tested on the following datasets:

- Cifar10
- Cifar10+ (with data augmentation)
- Cifar100
- Cifar100+ (with data augmentation)
- SVHN

The initial number of blocks and layers in each block (`layer_num_list`), the growth rate, image normalization and other training params may be changed trough shell or inside the source code.

Example run:

.. code::

    python run_dense_net.py --train --test --dataset=C10 -lnl '12,12,12'

List all available options:

.. code::

    python run_dense_net.py --help

Dependencies
------------

- Model was tested with Python 3.6.7 with CUDA
- Model should work as expected with TensorFlow >= 1.0.

Repo supported with requirements files - so the easiest way to install all just run:

- in case of CPU usage ``pip install -r requirements/cpu.txt``.
- in case of GPU usage ``pip install -r requirements/gpu.txt``.
