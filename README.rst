Self-constructing DenseNet with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Algorithm for automatically building `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`__ (DenseNets).

A modification of `Illarion Khlestov's TensorFlow implementation of DenseNets. <https://github.com/ikhlestov/vision_networks>`__

The algorithm that is currently implemented is based on the early stages of `the EMANN self-structuring algorithm by Salomé and Bersini (1994).
<https://ieeexplore.ieee.org/document/374473>`__
The algorithm is based on the evolution of two features, which are here called "connection strength" and "relevance for sources".

The **connection strength** (CS) of a layer :math:`l` with a previous layer :math:`s` (between the previous block's output and :math:`l-1`)
is the mean of the absolute kernel weights associated with the connection :math:`s-l` (sum of the weights divided by num of weights).
This value can also be presented as a normalised CS: the CS is then divided by the max CS for the layer :math:`l` to give it a value between 0 and 1.

The **relevance for sources** of a layer :math:`l` is also a value between 0 and 1.
It expresses how many of the connections received by :math:`l` are 'relevant enough' for their source layers to send information through them.
For each connection from a previous layer :math:`s` (between the previous block's output and :math:`l-1`), the relevance is increased by :math:`1/(l+1)`
if the connection's CS is >= 0.67 * the max CS for all connections sent by :math:`s`.

The most recent version of the self-constructing algorithm trains and builds the network in two stages:

- **Ascension stage:** a new layer is added to the last block every ``ascension_threshold`` epochs.
  The stage ends when at least one layer in the block has a 'relevance for sources' equal to 1.
  The ``ascension_threshold`` is a constant parameter with a default value of 10.

- **Improvement stage:** the algorithm begins a countdown of ``patience_parameter`` epochs before ending the stage.
  If however a layer settles during this stage, a new layer is added and the countdown is restarted.
  The ``patience_parameter`` is another constant parameter, its default value is 200.

After the completion of these two stages, the training ends. Experiments are currently being undertaken to implement a more efficient self-constructing algorithm.

In addition to the latest version, previous variants of the algorithm can be used by specifying parameters (see "Running the code" below).
The most recent variant is always the default version of the algorithm. The variants currently available are.

- **Variant #0:** Performs an ascension stage, then trains the resulting network until a max number of total epochs (default 300) has elapsed.

- **Variant #1:** Performs an ascension stage, then an improvement stage without using the restarting countdown system. The improvement stage lasts until a max number of total epochs (default 300) has elapsed.

- **Variant #2:** **This is the most recent version.** Performs an ascension stage, then an improvement stage that ends based on a restarting countdown system.

Running the code
----------------

Two types of DenseNets are available:

- DenseNet (without bottleneck layers): ``-m 'DenseNet'``
- DenseNet-BC (with bottleneck layers): ``-m 'DenseNet-BC'``

Each model can be tested on the following datasets:

- Cifar10:  ``--dataset=C10``
- Cifar10+:  ``--dataset=C10+`` (with data augmentation)
- Cifar100:  ``--dataset=C100``
- Cifar100+:  ``--dataset=C100+`` (with data augmentation)
- SVHN:  ``--dataset=SVHN``

**Example runs:**

``python run_dense_net.py --train --test -m 'DenseNet' --dataset=C10 --self-construct -at 10 -pp 200``

Here the program uses the self-constructing algorithm (most recent variant) to build a DenseNet trained on the Cifar10 dataset.
The resulting self-constructed network is then tested.
The ``ascension_threshold`` is set to 10, and the ``patience_parameter`` is set to 200.

``python run_dense_net.py --train --test -m 'DenseNet-BC' --dataset=C10+ --self-construct -var 1 -at 20 -pp 100``

Here the program uses the self-constructing algorithm (variant #1) to build a DenseNet-BC trained on the Cifar10 dataset, with data augmentation.
The resulting self-constructed network is then tested.
The ``ascension_threshold`` is set to 20, and the ``patience_parameter`` is set to 100.

``python run_dense_net.py --train --test -m 'DenseNet' --dataset=SVHN --no-self-construct -lnl '12,12,12'``

Here the program trains a prebuilt DenseNet on the SVHN dataset, and then tests it.
It does not run the self-constructing algorithm on it.
The prebuilt DenseNet has got 3 dense blocks, with 12 layers in each block.

**List all available options:**

``python run_dense_net.py --help``

**N.B.:** It is possible to specify (like in the second example run) a prebuilt initial structure on which the algorithm runs (different than one block with one layer).
A shell argument called ``layer_num_list`` (or ``lnl``) is used for defining the prebuilt blocks with their number of layers.

Example: ``-lnl '12,12,12'`` would mean that three blocks with 12 layers are created (the default architecture in Illarion Khlestov's implementation),
and the algorithm starts building inside the third block.

The default value for this parameter is ``'1'``. That is, the algorithm begins with only one block, containing a single layer, and starts adding new layers inside this block.

Other parameters such as the ``growth_rate`` (``k``), ``ascension_threshold`` (``at``), ``patience_parameter`` (``pp``), or image normalization methods,
may also be specified through shell arguments or by editing the source code (run_dense_net.py).

**N.B.2:** An alternative to 'relevance' is implemented in the code: 'spread of reception'.
It expresses how many of the connections received by :math:`l` are 'relevant enough' for l to receive information through them.
For each connection from a previous layer :math:`s` (between the previous block's output and :math:`l-1`), the relevance is increased by :math:`1/(l+1)`
if the connection's CS is >= 0.67 * the max CS for all connections received by :math:`l`.

An option for using 'spread' instead of 'relevance' in the algorithm will be implemented in the near future.

Dependencies
------------

- Model was tested with Python 3.6.7 with CUDA
- Model should work as expected with TensorFlow >= 1.0.

Repo supported with requirements files - to install them all just run:

- in case of CPU usage: ``pip install -r requirements/cpu.txt``.
- in case of GPU usage: ``pip install -r requirements/gpu.txt``.
