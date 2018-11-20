Self-constructing DenseNet with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Algorithm for automatically building `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`__ (DenseNets).

A modification of `Illarion Khlyestov's TensorFlow implementation of DenseNets. <https://github.com/ikhlestov/vision_networks>`__

Two types of DenseNets are available:

- DenseNet - without bottleneck layers
- DenseNet-BC - with bottleneck layers

Each model can be tested on the following datasets:

- Cifar10
- Cifar10+ (with data augmentation)
- Cifar100
- Cifar100+ (with data augmentation)
- SVHN

The initial number of blocks and layers in each block, the growth rate, image normalization and other training params may be changed trough shell or inside the source code.

Example run:

.. code::

    python run_dense_net.py --train --test --dataset=C10

List all available options:

.. code::

    python run_dense_net.py --help

Citation:

.. code::

     @article{Huang2016Densely,
            author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
            title = {Densely Connected Convolutional Networks},
            journal = {arXiv preprint arXiv:1608.06993},
            year = {2016}
     }

Dependencies
------------

- Model was tested with Python 3.6.7 with CUDA
- Model should work as expected with TensorFlow >= 1.0.

Repo supported with requirements files - so the easiest way to install all just run:

- in case of CPU usage ``pip install -r requirements/cpu.txt``.
- in case of GPU usage ``pip install -r requirements/gpu.txt``.
