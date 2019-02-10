import os
import time
import shutil
from datetime import timedelta, datetime

import numpy as np
import scipy.misc
import tensorflow as tf


TF_VERSION = list(map(int, tf.__version__.split('.')[:2]))


class DenseNet:

    # -------------------------------------------------------------------------
    # ----------------------------CLASS INITIALIZER----------------------------
    # -------------------------------------------------------------------------

    def __init__(self, data_provider, growth_rate, layer_num_list,
                 keep_prob, num_inter_threads, num_intra_threads,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_self_construct, should_save_logs,
                 feature_period, should_save_ft_logs,
                 check_kernel_features, measure_layer_cr_entr,
                 should_save_model, should_save_images,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 **kwargs):
        """
        Class to implement DenseNet networks as defined in this paper:
        https://arxiv.org/pdf/1611.05552.pdf

        Args:
            data_provider: data provider object for the required data set;
            growth_rate: `int`, number of new convolutions per dense layer;
            layer_num_list: `str`, list of number of layers in each block,
                separated by commas (e.g. '12,12,12');
            keep_prob: `float`, keep probability for dropout. If keep_prob = 1
                dropout will be disabled;
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4;
            nesterov_momentum: `float`, momentum for Nesterov optimizer;
            model_type: `str`, model type name ('DenseNet' or 'DenseNet-BC'),
                should we use bottleneck layers and compression or not;
            dataset: `str`, dataset name;
            should_self_construct: `bool`, should use self-constructing or not;
            should_save_logs: `bool`, should tensorflow logs be saved or not;
            feature_period: `int`, number of epochs between two measurements
                of feature values (e.g. accuracy, loss, kernel mean and std);
            should_save_ft_logs: `bool`, should feature logs be saved or not;
            measure_layer_cr_entr: `bool`, should measure cross-entropies for
                each individual layer in the last block or not;
            check_kernel_features: `bool`, should check kernel features or not;
            should_save_model: `bool`, should the model be saved or not;
            should_save_images: `bool`, should images be saved or not;
            renew_logs: `bool`, remove previous logs for current model;
            reduction: `float`, reduction (theta) at transition layers for
                DenseNets with compression (DenseNet-BC);
            bc_mode: `bool`, boolean equivalent of model_type, should we use
                bottleneck layers and compression (DenseNet-BC) or not.
        """
        self.creation_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        self.growth_rate = growth_rate
        self.num_inter_threads = num_inter_threads
        self.num_intra_threads = num_intra_threads
        # number of outputs (feature maps) produced by the initial convolution
        # (2*k, same value as in the original Torch code)
        self.first_output_features = growth_rate * 2
        self.layer_num_list = list(map(int, layer_num_list.split(',')))
        self.total_blocks = len(self.layer_num_list)
        self.bc_mode = bc_mode
        self.reduction = reduction

        print("Build %s model with %d blocks, "
              "The number of layers in each block is:" % (
                  model_type, self.total_blocks))
        if not bc_mode:
            print('\n'.join('Block %d: %d composite layers.' % (
                k, self.layer_num_list[k]) for k in range(len(
                    self.layer_num_list))))
        if bc_mode:
            self.layer_num_list[:] = [-(-l // 2) for l in self.layer_num_list]
            print('\n'.join('Block %d: %d bottleneck layers and %d composite'
                            'layers.' % (k, self.layer_num_list[k],
                                         self.layer_num_list[k])
                            for k in range(len(self.layer_num_list))))

        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.should_self_construct = should_self_construct
        self.should_save_logs = should_save_logs

        self.feature_period = feature_period
        self.should_save_ft_logs = should_save_ft_logs
        self.check_kernel_features = check_kernel_features
        self.measure_layer_cr_entr = measure_layer_cr_entr

        self.should_save_model = should_save_model
        self.should_save_images = should_save_images
        self.renew_logs = renew_logs
        self.batches_step = 0

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    # -------------------------------------------------------------------------
    # ------------------------SAVING AND LOADING DATA -------------------------
    # -------------------------------------------------------------------------

    def update_paths(self):
        """
        Update all paths for saving data to their proper values.
        This is used after the graph is modified (new block or layer).
        This is also used after an AttributeError when calling these paths.
        """
        save_path = 'saves/%s' % self.model_identifier
        if self.should_save_model:
            os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, 'model.chkpt')
        self._save_path = save_path

        logs_path = 'logs/%s' % self.model_identifier
        if self.should_save_logs:
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
        self._logs_path = logs_path

        ft_logs_path = 'ft_logs/%s' % self.run_identifier
        if self.should_save_ft_logs:
            os.makedirs('ft_logs/', exist_ok=True)
        self._ft_logs_path = ft_logs_path

        images_path = 'images/%s' % self.run_identifier
        if self.should_save_images:
            os.makedirs(images_path, exist_ok=True)
        self._images_path = images_path

        return save_path, logs_path, ft_logs_path, images_path

    @property
    def model_identifier(self):
        """
        Returns an identifier `str` for the current DenseNet model.
        It gives the model's type ('DenseNet' or 'DenseNet-BC'),
        its growth rate k, the number of layers in each block,
        and the dataset that was used.
        """
        return "{}_growth_rate={}_layer_num_list={}_dataset_{}".format(
            self.model_type, self.growth_rate, ",".join(map(
                str, self.layer_num_list)), self.dataset_name)

    @property
    def run_identifier(self):
        """
        Returns an identifier `str` for the current execution of the algorithm.
        It gives the model's type ('DenseNet' or 'DenseNet-BC'),
        its growth rate k, the dataset that was used,
        and the date and hour at which the execution started.
        """
        return "{}_{}_growth_rate={}_dataset_{}".format(
            self.model_type, self.creation_time, self.growth_rate,
            self.dataset_name)

    @property
    def save_path(self):
        """
        Returns a path where the saver should save the current model.
        """
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = self.update_paths()[0]
        return save_path

    @property
    def logs_path(self):
        """
        Returns a path where the logs for the current model should be written.
        """
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = self.update_paths()[1]
        return logs_path

    @property
    def ft_logs_path(self):
        """
        Returns a path where the evolution of features in the current execution
        should be recorded.
        """
        try:
            ft_logs_path = self._ft_logs_path
        except AttributeError:
            ft_logs_path = self.update_paths()[2]
        return ft_logs_path

    @property
    def images_path(self):
        """
        Returns a path where images from the current execution should be saved.
        """
        try:
            images_path = self._images_path
        except AttributeError:
            images_path = self.update_paths()[3]
        return images_path

    def save_model(self, global_step=None):
        """
        Saves the current trained model at the proper path, using the saver.

        Args:
            global_step: `int` or None, used for numbering saved model files
        """
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        """
        Loads a saved model to use (instead of a new one) using the saver.
        This is a previously trained and saved model using the model_type
        ('DenseNet' or 'DenseNet-BC'), growth rate, layers in each block,
        and dataset that was specified in the program arguments.
        """
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        """
        Writes a log of the current mean loss (cross_entropy) and accuracy.

        Args:
            loss: `float`, loss (cross_entropy) for the current log
            accuracy: `float`, accuracy for the current log
            epoch: `int`, current training epoch (or batch)
            prefix: `str`, is this log for a batch ('per_batch'), a
                training epoch ('train') or a validation epoch ('valid')
            should_print: `bool`, should we print this log on console or not
        """
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def process_kernels(self, kernel, block_num, kernel_num, epoch):
        """
        Process the kernels of a given convolution filter, in some cases record
        their mean and standard deviation values in the feature logs, and/or
        save a representation of the kernels PNG image file.

        Args:
            kernel: tensor, the kernel to save
            block_num: `int`, identifier number for the kernel's block
            kernel_num: `int`, identifier for the kernel within the block
            epoch: `int`, current training epoch (or batch)
        """
        # get an array representation of the kernels, then get its dimensions
        k_image = self.sess.run(kernel)
        k_d = kernel.get_shape().as_list()
        print('* Block %d kernel %d: mean = %f, std = %f' % (
            block_num, kernel_num, np.mean(k_image), np.std(k_image)
        ))

        if self.should_save_ft_logs:
            # save feature logs for the kernel mean and standard deviation
            self.feature_writer.write((';\"%f\";\"%f\"' % (
                np.mean(k_image), np.std(k_image))).replace(".", ","))

        if self.should_save_images:
            # properly format the kernels to save them as an image
            k_image = k_image.transpose()
            k_image = np.moveaxis(k_image, [1, 2], [2, 1])
            k_image = np.resize(k_image, (k_d[1]*k_d[3], k_d[0]*k_d[2]))

            # save the image in the proper file
            im_filepath = './%s/block_%d_kernel_%d' % (
                self.images_path, block_num, kernel_num)
            os.makedirs(im_filepath, exist_ok=True)
            im_filepath += '/epoch_%d.png' % epoch
            scipy.misc.imsave(im_filepath, k_image)

    # -------------------------------------------------------------------------
    # -----------------------DEFINING INPUT PLACEHOLDERS-----------------------
    # -------------------------------------------------------------------------

    def _define_inputs(self):
        """
        Defines some imput placeholder tensors:
        images, labels, learning_rate, is_training.
        """
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    # -------------------------------------------------------------------------
    # -----------------------BUILDING THE DENSENET GRAPH-----------------------
    # -------------------------------------------------------------------------

    # SIMPLEST OPERATIONS -----------------------------------------------------
    # -------------------------------------------------------------------------

    def weight_variable_msra(self, shape, name):
        """
        Creates weights for a fully-connected layer, using an initialization
        method which does not scale the variance.

        Args:
            shape: `list` of `int`, shape of the weight matrix
            name: `str`, a name for identifying the weight matrix
        """
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def avg_pool(self, _input, k):
        """
        Performs average pooling on a given input (_input),
        within square kernels of side k and stride k.

        Args:
            _input: tensor, the operation's input
            k: `int`, the size and stride for the kernels
        """
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        """
        Performs batch normalization on a given input (_input).

        Args:
            _input: tensor, the operation's input
        """
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        """
        Creates a 2D convolutional layer (applies a certain number of
        kernels on some input features to obtain output features).
        Returns the output of the layer and a reference to its kernel.

        Args:
            _input: tensor, the operation's input
            out_features: `int`, number of feature maps at the output
            kernel_size: `int`, size of the square kernels (their side)
            strides: `list` of `int`, strides in each direction for kernels
            padding: `str`, should we use padding ('SAME') or not ('VALID')
        """
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output, kernel

    def dropout(self, _input):
        """
        If the given keep_prob is not 1 AND if the graph is being trained,
        performs a random dropout operation on a given input (_input).
        The dropout probability is the keep_prob parameter.

        Args:
            _input: tensor, the operation's input
        """
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    # SIMPLEST OPERATIONS (FULLY CONNECTED)------------------------------------
    # -------------------------------------------------------------------------

    def weight_variable_xavier(self, shape, name):
        """
        Creates weights for a fully-connected layer, using the Xavier
        initializer (keeps gradient scale roughly the same in all layers).

        Args:
            shape: `list` of `int`, shape of the weight matrix
            name: `str`, a name for identifying the weight matrix
        """
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        """
        Creates bias terms for a fully-connected layer, initialized to 0.0.

        Args:
            shape: `list` of `int`, shape of the bias matrix
            name: `str`, a name for identifying the bias matrix
        """
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    # COMPOSITE FUNCTION + BOTTLENECK -----------------------------------------
    # -------------------------------------------------------------------------

    def composite_function(self, _input, out_features, kernel_size=3):
        """
        Function H_l. Takes a concatenation of previous outputs and performs:
        - batch normalization
        - ReLU activation function
        - 2d convolution, with required kernel size (side)
        - dropout, if required (training the graph and keep_prob not set to 1)
        Returns the output tensor and a reference to the 2d convolution kernel.

        Args:
            _input: tensor, the operation's input
            out_features: `int`, number of feature maps at the output
            kernel_size: `int`, size of the square kernels (their side)
        """
        with tf.variable_scope("composite_function"):
            # batch normalization
            output = self.batch_norm(_input)
            # ReLU activation function
            output = tf.nn.relu(output)
            # 2d convolution
            output, kernel_ref = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            # dropout (if the graph is being trained and keep_prob is not 1)
            output = self.dropout(output)
        return output, kernel_ref

    def bottleneck(self, _input, out_features):
        """
        Bottleneck function, used before composite function H_l in DenseNet-BC,
        takes a concatenation of previous outputs and performs:
        - batch normalization
        - ReLU activation function
        - 2d convolution, with kernel size 1 (produces 4x the features of H_l)
        - dropout, if required (training the graph and keep_prob not set to 1)
        Returns the output tensor and a reference to the 2d convolution kernel.

        Args:
            _input: tensor, the operation's input
            out_features: `int`, number of feature maps at the output of H_l
            kernel_size: `int`, size of the square kernels (their side)
        """
        with tf.variable_scope("bottleneck"):
            # batch normalization
            output = self.batch_norm(_input)
            # ReLU activation function
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            # 2d convolution (produces intermediate features)
            output, kernel_ref = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            # dropout (if the graph is being trained and keep_prob is not 1)
            output = self.dropout(output)
        return output, kernel_ref

    # BLOCKS AND THEIR INTERNAL LAYERS ----------------------------------------
    # -------------------------------------------------------------------------

    def add_internal_layer(self, _input, layer, growth_rate):
        """
        Adds a new convolutional (dense) layer within a block.
        This layer will perform the composite function H_l([x_0, ..., x_l-1])
        to obtain its output x_l.
        It will then concatenate x_l with the layer's input: all the outputs of
        the previous layers, resulting in [x_0, ..., x_l-1, x_l].

        Args:
            _input: tensor, the operation's input
            layer: `int`, identifier number for this layer (within a block)
            growth_rate: `int`, number of new convolutions per dense layer
        """
        with tf.variable_scope("layer_%d" % layer):
            # use the composite function H_l (3x3 kernel conv)
            if not self.bc_mode:
                comp_out, kernel_ref = self.composite_function(
                    _input, out_features=growth_rate, kernel_size=3)
            # in DenseNet-BC mode, add a bottleneck layer before H_l (1x1 conv)
            elif self.bc_mode:
                bottleneck_out, kernel_ref = self.bottleneck(
                    _input, out_features=growth_rate)
                if self.check_kernel_features:
                    self.kernel_ref_list[-1].append(kernel_ref)
                comp_out, kernel_ref = self.composite_function(
                    bottleneck_out, out_features=growth_rate, kernel_size=3)
            # save a reference to the composite function's kernel
            if self.check_kernel_features:
                self.kernel_ref_list[-1].append(kernel_ref)
            # concatenate output of H_l with layer input (all previous outputs)
            if TF_VERSION[0] >= 1 and TF_VERSION[1] >= 0:
                output = tf.concat(axis=3, values=(_input, comp_out))
            else:
                output = tf.concat(3, (_input, comp_out))
        return output

    def add_block(self, _input, block, growth_rate, layers_in_block, is_last):
        """
        Adds a new block containing several convolutional (dense) layers.
        These are connected together following a DenseNet architecture,
        as defined in the paper.

        Args:
            _input: tensor, the operation's input
            block: `int`, identifier number for this block
            growth_rate: `int`, number of new convolutions per dense layer
            layers_in_block: `int`, number of dense layers in this block
            is_last: `bool`, is this the last block in the network or not
        """
        if self.check_kernel_features:
            self.kernel_ref_list.append([])
        if is_last:
            self.cross_entropy = []

        with tf.variable_scope("Block_%d" % block) as self.current_block:
            output = _input
            for layer in range(layers_in_block):
                output = self.add_internal_layer(output, layer, growth_rate)

                if is_last and self.measure_layer_cr_entr:
                    # Save the cross-entropy for all layers except the last one
                    # (it is always saved as part of the end-graph operations)
                    if layer != layers_in_block-1:
                        _, cross_entropy = self.cross_entropy_loss(
                            output, self.labels, block, layer)
                        self.cross_entropy.append(cross_entropy)
        return output

    # TRANSITION LAYERS -------------------------------------------------------
    # -------------------------------------------------------------------------

    def transition_layer(self, _input, block):
        """
        Adds a new transition layer after a block. This layer's inputs are the
        concatenated feature maps of each layer in the block.
        The layer first runs the composite function with kernel size 1:
        - In DenseNet mode, it produces as many feature maps as the input had.
        - In DenseNet-BC mode, it produces reduction (theta) times as many,
          compressing the output.
        Afterwards, an average pooling operation (of size 2) is carried to
        change the output's size.

        Args:
            _input: tensor, the operation's input
            block: `int`, identifier number for the previous block
        """
        with tf.variable_scope("Transition_after_block_%d" % block):
            # add feature map compression in DenseNet-BC mode
            out_features = int(int(_input.get_shape()[-1]) * self.reduction)
            # use the composite function H_l (1x1 kernel conv)
            output, kernel_ref = self.composite_function(
                _input, out_features=out_features, kernel_size=1)
            # save a reference to the composite function's kernel
            if self.check_kernel_features:
                self.kernel_ref_list[-1].append(kernel_ref)
            # use average pooling to reduce feature map size
            output = self.avg_pool(output, k=2)
        return output

    def transition_layer_to_classes(self, _input, block, layer):
        """
        Adds the transition layer after the last block. This layer outputs the
        estimated probabilities by classes. It performs:
        - batch normalization
        - ReLU activation function
        - wider-than-normal average pooling
        - reshaping the output into a 1D tensor
        - fully-connected layer (matrix multiplication, weights and biases)

        Args:
            _input: tensor, the operation's input
            block: `int`, identifier number for the last block
            layer: `int`, identifier number for the last layer in that block
        """
        with tf.variable_scope("Transition_to_classes_block_%d_layer_%d" %
                               (block, layer)):
            # batch normalization
            output = self.batch_norm(_input)
            # ReLU activation function
            output = tf.nn.relu(output)
            # wide average pooling
            last_pool_kernel = int(output.get_shape()[-2])
            output = self.avg_pool(output, k=last_pool_kernel)
            # reshaping the output into 1D
            features_total = int(output.get_shape()[-1])
            output = tf.reshape(output, [-1, features_total])
            # fully-connected layer
            W = self.weight_variable_xavier(
                [features_total, self.n_classes], name='W')
            bias = self.bias_variable([self.n_classes])
            logits = tf.matmul(output, W) + bias
        return logits

    # END GRAPH OPERATIONS ----------------------------------------------------
    # -------------------------------------------------------------------------

    def cross_entropy_loss(self, _input, labels, block, layer):
        """
        Takes an input and adds a transition layer to obtain predictions for
        classes. Then calculates the cross-entropy loss for that input with
        respect to expected labels. Returns the prediction tensor and the
        calculated cross-entropy.

        Args:
            _input: tensor, the operation's input
            labels: tensor, the expected labels (classes) for the data
            block: `int`, identifier number for the last block
            layer: `int`, identifier number for the last layer in that block
        """
        # add the FC transition layer to the classes (+ softmax).
        logits = self.transition_layer_to_classes(_input, block, layer)
        prediction = tf.nn.softmax(logits)

        # set the calculation for the losses (cross_entropy and l2_loss)
        if TF_VERSION[0] >= 1 and TF_VERSION[1] >= 5:
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                           labels=labels))
        else:
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=labels))

        return prediction, cross_entropy

    def _define_end_graph_operations(self):
        """
        Adds the last layer on top of the (editable portion of the) graph.
        Then defines the operations for cross-entropy, the training step,
        and the accuracy.
        """
        # obtain the predicted logits, set the calculation for the losses
        # (cross_entropy and l2_loss)
        prediction, cross_entropy = self.cross_entropy_loss(
            self.output, self.labels, self.total_blocks-1,
            self.layer_num_list[-1]-1)
        self.cross_entropy.append(cross_entropy)
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # set the optimizer and define the training step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(
            cross_entropy + l2_loss * self.weight_decay)

        # set the calculation for the accuracy
        correct_prediction = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # MAIN GRAPH BUILDING FUNCTIONS -------------------------------------------
    # -------------------------------------------------------------------------

    def _new_layer(self):
        """
        Add a new layer at the end of the current last block.
        In DenseNet-BC mode, two layers (bottleneck and compression) will be
        added instead of just one.
        """
        # safely access the current block's variable scope
        with tf.variable_scope(self.current_block,
                               auxiliary_name_scope=False) as cblock_scope:
            with tf.name_scope(cblock_scope.original_name_scope):
                self.output = self.add_internal_layer(
                    self.output, self.layer_num_list[-1], self.growth_rate)
        self.layer_num_list[-1] += 1

        # Refresh the cross-entropy list if not measuring layer cross-entropies
        if not self.measure_layer_cr_entr:
            self.cross_entropy = []

        if not self.bc_mode:
            print("ADDED A NEW LAYER to the last block (#%d)! "
                  "It now has got %d layers" %
                  (self.total_blocks-1, self.layer_num_list[-1]))
        if self.bc_mode:
            print("ADDED A NEW PAIR OF LAYERS to the last block (#%d)! "
                  "It now has got %d bottleneck and composite layers" %
                  (self.total_blocks-1, self.layer_num_list[-1]))

        self.update_paths()
        self._define_end_graph_operations()
        self._initialize_uninitialized_variables()

    def _new_block(self):
        """
        Add a transition layer, and a new block (with one layer) at the end
        of the current last block.
        In DenseNet-BC mode, the new module will begin with two layers
        (bottleneck and compression) instead of just one.
        """
        self.output = self.transition_layer(self.output, self.total_blocks-1)
        self.output = self.add_block(
            self.output, self.total_blocks, self.growth_rate, 1, True)
        self.layer_num_list.append(1)
        self.total_blocks += 1

        print("ADDED A NEW BLOCK (#%d), "
              "The number of layers in each block is now:" %
              (self.total_blocks-1))
        if not self.bc_mode:
            print('\n'.join('Block %d: %d composite layers.' % (
                k, self.layer_num_list[k]) for k in range(len(
                    self.layer_num_list))))
        if self.bc_mode:
            print('\n'.join('Block %d: %d bottleneck layers and %d composite'
                            'layers.' % (k, self.layer_num_list[k],
                                         self.layer_num_list[k])
                            for k in range(len(self.layer_num_list))))

        self.update_paths()
        self._define_end_graph_operations()
        self._initialize_uninitialized_variables()

    def _build_graph(self):
        """
        Gets the growth rate and the layers per block.
        Then builds the graph and defines the operations for:
        cross-entropy (also l2_loss and a momentum optimizer),
        training step (minimize momentum optimizer using l2_loss + cross-entr),
        accuracy (reduce mean).
        """
        growth_rate = self.growth_rate
        layers_in_each_block = self.layer_num_list
        self.output = self.images

        # first add a 3x3 convolution layer with first_output_features outputs
        with tf.variable_scope("Initial_convolution"):
            self.output, kernel_ref = self.conv2d(
                self.output, out_features=self.first_output_features,
                kernel_size=3)
            if self.check_kernel_features:
                self.kernel_ref_list = [[kernel_ref]]

        # then add the required blocks
        for block in range(self.total_blocks):
            self.output = self.add_block(
                self.output, block, growth_rate, layers_in_each_block[block],
                block == self.total_blocks - 1)
            #  all blocks except the last have transition layers
            if block != self.total_blocks - 1:
                self.output = self.transition_layer(self.output, block)

        self._define_end_graph_operations()

    # -------------------------------------------------------------------------
    # -------------------INITIALIZING THE TENSORFLOW SESSION-------------------
    # -------------------------------------------------------------------------

    def _initialize_uninitialized_variables(self):
        """
        Finds the references to all uninitialized variables, then tells
        TensorFlow to initialize these variables.
        """
        # get a set with all the names of uninitialized variables
        uninit_varnames = list(map(str, self.sess.run(
            tf.report_uninitialized_variables())))
        uninit_vars = []
        # for every variable, check if its name is in the uninitialized set
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            varname = 'b\'' + var.name.split(':')[0] + '\''
            if varname in uninit_varnames:
                uninit_vars.append(var)
        # initialize all the new variables
        self.sess.run(tf.variables_initializer(uninit_vars))

    def _initialize_all_variables(self):
        """
        Tells TensorFlow to initialize all variables, using the proper method
        for the TensorFlow version.
        """
        if TF_VERSION[0] >= 0 and TF_VERSION[1] >= 10:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess.run(tf.initialize_all_variables())

    def _initialize_session(self):
        """
        Starts a TensorFlow session with the correct configuration.
        Then tells TensorFlow to initialize all variables, create a saver
        and a log file writer.
        """
        config = tf.ConfigProto()

        # specify the CPU inter and intra threads used by MKL
        config.intra_op_parallelism_threads = self.num_intra_threads
        config.inter_op_parallelism_threads = self.num_inter_threads

        # restrict model GPU memory utilization to the minimum required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # initialize variables, create saver, create log file writers
        self._initialize_all_variables()
        self.saver = tf.train.Saver()
        if self.should_save_logs:
            if TF_VERSION[0] >= 0 and TF_VERSION[1] >= 10:
                logswriter = tf.summary.FileWriter
            else:
                logswriter = tf.train.SummaryWriter
            self.summary_writer = logswriter(self.logs_path)
        if self.should_save_ft_logs:
            self.feature_writer = open('./%s.csv' % self.ft_logs_path, "w")

    # -------------------------------------------------------------------------
    # --------------------COUNTING ALL TRAINABLE PARAMETERS--------------------
    # -------------------------------------------------------------------------

    def _count_trainable_params(self):
        """
        Uses TensorFlow commands to count the number of trainable parameters
        in the graph (sum of the multiplied dimensions of each TF variable).
        Then prints the number of parameters.
        """
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total trainable params: %.1fk" % (total_parameters / 1e3))

    # -------------------------------------------------------------------------
    # ---------------------TRAINING AND TESTING THE MODEL----------------------
    # -------------------------------------------------------------------------

    def print_relevant_features(self, loss, accuracy, epoch):
        """
        Prints on console the current values of relevant features.
        If feature logs are being saved, this function also saves these values.
        If images are being saved, kernel states are also saved as images.

        Args:
            loss: `list` of `float`, validation set loss (cross_entropy) for
                this epoch, corresponding to each internal layer of the graph
            accuracy: `float`, validation set accuracy for this epoch
            epoch: `int`, current training epoch
        """
        # print the current accuracy and the cross-entropy for each layer
        print("Current validation accuracy = %f" % accuracy)
        print("Cross-entropy per layer in block #%d:" % (
            self.total_blocks-1))
        for l in range(len(loss)):
            print("* Layer #%d: cross-entropy = %f" % (l, loss[l]))

        if self.should_save_ft_logs:
            # save the previously printed feature values
            self.feature_writer.write(("\"Epoch %d\";\"%f\";" % (
                epoch, accuracy)).replace(".", ","))
            for l in range(len(loss)):
                self.feature_writer.write(("\"%f\";" % loss[l]
                                           ).replace(".", ","))
            self.feature_writer.write('\"\"')

        if self.check_kernel_features:
            # process kernels, sometimes save images and/or their mean and std
            print('-' * 40 + "\nProcessing kernels:")
            for b in range(-1, self.total_blocks):
                for k in range(len(self.kernel_ref_list[b+1])):
                    self.process_kernels(
                        self.kernel_ref_list[b+1][k], b, k, epoch)

        print('-' * 40)
        if self.should_save_ft_logs:
            self.feature_writer.write('\n')

    def self_constructing_epoch(self, epoch):
        """
        The self-constructing step for one training epoch. Adds new layers
        and/or blocks depending on parameters.
        Returns True if training should continue, False otherwise.

        Args:
            epoch: `int`, current training epoch
        """
        # naive W.I.P. self-constructing algorithm
        # useful to test the effects of adding layers/blocks at certain moments
        continue_training = True
        if epoch != 1 and epoch <= 40:
            # if (epoch-1) % 20 == 0:
            #     self._new_block()
            if (epoch-1) % 10 == 0:
                self._new_layer()

        return continue_training

    def train_one_epoch(self, data, batch_size, learning_rate):
        """
        Trains the model for one epoch using data from the proper training set.

        Args:
            data: training data yielded by the dataset's data provider
            batch_size: `int`, number of examples in a training batch
            learning_rate: `int`, learning rate for the optimizer
        """
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []

        # save each training batch's loss and accuracy
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            images, labels = batch
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True,
            }
            fetches = [self.train_step, self.cross_entropy[-1], self.accuracy]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy = result
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(
                    loss, accuracy, self.batches_step, prefix='per_batch',
                    should_print=False)

        # use the saved data to calculate the mean loss and accuracy
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def test(self, data, batch_size):
        """
        Tests the model using the proper testing set.

        Args:
            data: testing data yielded by the dataset's data provider
            batch_size: `int`, number of examples in a testing batch
        """
        num_examples = data.num_examples
        total_loss = []
        for l in range(len(self.cross_entropy)):
            total_loss.append([])
        total_accuracy = []

        # save each testing batch's loss and accuracy
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            feed_dict = {
                self.images: batch[0],
                self.labels: batch[1],
                self.is_training: False,
            }
            loss = self.sess.run(self.cross_entropy, feed_dict=feed_dict)
            accuracy = self.sess.run(self.accuracy, feed_dict=feed_dict)
            for j in range(len(loss)):
                total_loss[j].append(loss[j])
            total_accuracy.append(accuracy)

        # use the saved data to calculate the mean loss and accuracy
        mean_loss = []
        for loss_list in total_loss:
            mean_loss.append(np.mean(loss_list))
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def train_all_epochs(self, train_params):
        """
        Trains the model for a certain number of epochs, using parameters
        specified in the train_params argument.

        Args (in train_params):
            batch_size: `int`, number of examples in a training batch
            max_n_epochs: `int`, maximum number of training epochs to run
            initial_learning_rate: `int`, initial learning rate for optimizer
            reduce_lr_epoch_1: `int`, if not self-constructing the network,
                first epoch where the current learning rate is divided by 10
                (initial_learning_rate/10)
            reduce_lr_epoch_2: `int`, if not self-constructing the network,
                second epoch where the current learning rate is divided by 10
                (initial_learning_rate/100)
            validation_set: `bool`, should a validation set be used or not
            validation_split: `int` or None
                `float`: chunk of the training set used as the validation set
                None: use the testing set as the validation set
            shuffle: `str` or None, or `bool`
                `str` or None: used with CIFAR datasets, should we shuffle the
                data only before training ('once_prior_train'), on every epoch
                ('every_epoch') or not at all (None)
                `bool`: used with SVHN, should we shuffle the data or not
            normalization: `str` or None
                None: don't use any normalization for pixels
                'divide_255': divide all pixels by 255
                'divide_256': divide all pixels by 256
                'by_chanels': substract the mean of the pixel's chanel and
                    divide the result by the channel's standard deviation
        """
        max_n_epochs = train_params['max_n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()

        epoch = 1
        while epoch < max_n_epochs + 1:
            # only print epoch name on certain epochs
            if (epoch-1) % self.feature_period == 0:
                print('\n', '-'*30, "Train epoch: %d" % epoch, '-'*30, '\n')
            start_time = time.time()

            # learning rate only decreases when not self-constructing
            if not self.should_self_construct:
                if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                    learning_rate = learning_rate / 10
                    print("Decrease learning rate, new lr = %f" %
                          learning_rate)

            # training step for one epoch
            print("Training...", end=' ')
            loss, acc = self.train_one_epoch(
                self.data_provider.train, batch_size, learning_rate)
            if self.should_save_logs:
                self.log_loss_accuracy(loss, acc, epoch, prefix='train')

            # validation step after the epoch
            if train_params.get('validation_set', False):
                print("Validation...")
                loss, acc = self.test(
                    self.data_provider.validation, batch_size)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss[-1], acc, epoch,
                                           prefix='valid')

            # on certain epochs print (and perhaps save) feature values
            if (epoch-1) % self.feature_period == 0:
                self.print_relevant_features(loss, acc, epoch)

            # self-constructing step
            if self.should_self_construct:
                if not self.self_constructing_epoch(epoch):
                    break

            # measure training time for this epoch
            time_per_epoch = time.time() - start_time
            seconds_left = int((max_n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if self.should_save_model:
                self.save_model()
            epoch += 1

        # measure total training time
        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))
