from BasePredictor import BasePredictor
import mxnet as mx

import numpy as np
import rna

import theano
import theano.tensor as T

import lasagne


class NaivePredictor(BasePredictor):
    """Naive approach to RNA secondary structure prediction based on MLP multi-class classification."""

    def __init__(self, sequence_length, substrings=False, max_examples=500, library='mxnet', data_model='linear'):
        """Make Naive Predictor.

        Args:
            sequence_length: Length of the sequence whose structure is predicted.
            substrings (bool): Accept sequence substrings for training phase.
            max_examples: Limit of examples predictor trains on.
            library: Underlying library ('mxnet' or 'lasagne') to make network topology and train on it.
            data_model: Model ('linear' or 'matrix') of data that is the input to the network.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.max_examples = max_examples
        self.substrings = substrings
        self.library = library
        self.data_model = data_model

    def preprocess(self):
        """Preprocess loaded data.

        Returns:
            X, y: NDArray of sequences and NDArray of labels.
        """
        X = self.X
        y = []
        list = []
        for i in X:
            if self.substrings:
                m = rna.Molecule(i[0, 0], i[0, 1])
                for j in m.get_substrings(self.sequence_length):
                    seq = j.seq
                    dot = j.dot
                    if rna.dot_reverse(dot) in y:
                        seq = seq[::-1]
                        dot = rna.dot_reverse(dot)
                    list.append(rna.encode_rna(seq))
                    y.append(dot)
                    # list.append(rna.encode_rna(j.seq))
                    # y.append(j.dot)
            else:
                if len(i[0, 0]) == self.sequence_length:
                    seq = i[0, 0]
                    dot = i[0, 1]
                    if rna.dot_reverse(dot) in y:
                        seq = seq[::-1]
                        dot = rna.dot_reverse(dot)
                    if self.data_model == 'linear':
                        list.append(rna.encode_rna(seq))
                    elif self.data_model == 'matrix':
                        list.append(rna.complementarity_matrix(rna.Molecule(seq)))
                    y.append(dot)
        X = np.array(list)
        y = y[:self.max_examples]
        z = set(y)
        self.num_labels = len(z)
        self.a = {}
        idx = 0
        for i in z:
            self.a[i] = idx
            idx += 1
        for i in range(len(y)):
            y[i] = self.a[y[i]]
        y = np.array(y)
        return X[:self.max_examples, :], y[:self.max_examples]

    def train_X(self):
        """Train the predictor with already saved data."""
        X, y = self.preprocess()

        if self.library == 'mxnet':
            data = mx.sym.Variable('data')

            fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=self.num_labels * 10)
            act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

            # The second fully-connected layer and the according activation function
            fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=self.num_labels * 5)
            act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")

            # The thrid fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
            fc4 = mx.sym.FullyConnected(data=act2, name='fc4', num_hidden=self.num_labels)
            # The softmax and loss layer
            mlp = mx.sym.SoftmaxOutput(data=fc4, name='softmax')
            # create a model
            # mx.viz.plot_network(symbol=mlp, shape={"data": (28, 22)}).render("NaiveNet", view=True)
            examples = mx.io.NDArrayIter(X, y)

            import logging
            logging.basicConfig(level=logging.INFO)
            self.model = mx.model.FeedForward(symbol=mlp,
                                              num_epoch=350,
                                              learning_rate=0.001,
                                              wd=0.00001,
                                              momentum=0.9)

            self.model.fit(X=examples)
        if self.library == 'lasagne':
            if self.data_model == 'linear':
                input_var = T.matrix('inputs')
            elif self.data_model == 'matrix':
                input_var = T.tensor3('inputs')
            target_var = T.ivector('targets')

            shape = (None, self.sequence_length)
            if self.data_model == 'matrix':
                shape = (None, self.sequence_length, self.sequence_length)

            l_in = lasagne.layers.InputLayer(shape=shape,
                                             input_var=input_var)
            l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
            l_hid1 = lasagne.layers.DenseLayer(
                l_in_drop, num_units=800,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
            l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

            l_hid2 = lasagne.layers.DenseLayer(
                l_hid1_drop, num_units=800,
                nonlinearity=lasagne.nonlinearities.rectify)

            l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)
            l_out = lasagne.layers.DenseLayer(
                l_hid2_drop, num_units=self.num_labels,
                nonlinearity=lasagne.nonlinearities.softmax)

            prediction = lasagne.layers.get_output(l_out)
            loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
            params = lasagne.layers.get_all_params(l_out, trainable=True)
            updates = lasagne.updates.sgd(loss, params, learning_rate=0.01)

            f_learn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
            self.model = theano.function([input_var], prediction, allow_input_downcast=True)

            # Training
            it = 5000
            for i in range(it):
                l = f_learn(X, y)

    def predict(self, seq):
        """Predict the secondary structure of RNA sequence.

        Args:
            seq: RNA sequence.

        Returns:
            m: Molecule object with predicted bracket notation.
        """
        prob = [[], []]
        dot = ''
        if self.library == 'mxnet':
            example = mx.io.NDArrayIter(np.array([rna.encode_rna(seq), rna.encode_rna(seq[::-1])]))
            prob = self.model.predict(example)

        if self.library == 'lasagne':
            if self.data_model == 'linear':
                prob = self.model(np.array([rna.encode_rna(seq), rna.encode_rna(seq[::-1])]))
            elif self.data_model == 'matrix':
                prob = self.model(np.array([rna.complementarity_matrix(rna.Molecule(seq)),
                                            rna.complementarity_matrix(rna.Molecule(seq[::-1]))]))
        backwards = False
        if prob[0].max() > prob[1].max():
            max = prob[0].argmax()
        else:
            max = prob[1].argmax()
            backwards = True
        for i, j in self.a.items():
            if j == max:
                dot = i
                break
        if backwards:
            dot = rna.dot_reverse(dot)
        m = rna.Molecule(seq, dot)
        return m
