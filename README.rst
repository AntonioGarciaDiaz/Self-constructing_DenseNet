Self-constructing DenseNet with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Algorithm for automatically building `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`__ (DenseNets).

A modification of `Illarion Khlestov's TensorFlow implementation of DenseNets. <https://github.com/ikhlestov/vision_networks>`__

The algorithm that is currently implemented is rather naive: it adds 1 new layer every 10 epochs and 1 new block every 20 epochs, until it reaches epoch 40. Naive algorithms like this one will always produce the same structure if stopped at a fixed epoch, which is very useful for measuring the inpact of adding elements to the topology at specific stages of the training stage. Experiments are currently being undertaken to implement a proper self-constructing algorithm in the future.

N.B.: For these self-constructing algorithms, I've chosen to implement a more "flexible" definition of DenseNets, where not all blocks necessarily have the same ammount of layers.
This means that instead of the usual parameters for the number of layers (depth) and blocks, a parameter called :code:`layer_num_list` is used for defining the initial blocks with their number of layers.

Example: :code:`--layer_num_list '12,12,12'` would mean that three blocks with 12 layers are created (the default architecture in Illarion Khlestov's implementation), and the algorithm starts building inside the third block.

The default value for this parameter is :code:`'1'`. That is, the algorithm begins with one block, containing a single layer, and can build new layers inside this block.

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
