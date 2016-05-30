import numpy as np
import theano
import theano.tensor as tt
import theano.tensor.nnet.conv as conv
from theano.tensor.signal.pool import pool_2d


RNG = np.random.RandomState(1234)


class DenseLayer(object):

    def __init__(self, input, in_size, out_size, W_values=None, b_values=None,
                 activation=tt.tanh):
        self.input = input

        if W_values is None:
            W_values = self._init((in_size, out_size))
        if b_values is None:
            b_values = np.zeros((out_size,), dtype=theano.config.floatX)

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        
        lin_output = tt.dot(self.input, self.W) + self.b
        self.output = (lin_output if activation == None
                       else activation(lin_output))
        self.params = [self.W, self.b]

    def _init(self, shape):
        W_bound = np.sqrt(6. / (shape[0] + shape[1]))
        return np.asarray(
            RNG.uniform(low=-W_bound, high=W_bound, size=shape),
            dtype=theano.config.floatX
        )


class ConvPoolLayer(object):

    def __init__(self, input, filter_shape, image_shape,
                 W_values, b_values, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = tt.tanh(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        )
        self.params = [self.W, self.b]

    def _init_W(self, shape):
        fan_in = np.prod(shape[1:])
        fan_out = (shape[0] * np.prod(shape[2:]) / 4)

        W_bound = np.sqrt(6. / (fan_in + fan_out))
        return np.asarray(
            RNG.uniform(low=-W_bound, high=W_bound, size=shape),
            dtype=theano.config.floatX
        )


class LSTMLayer(object):

    def __init__(self, input, in_size, out_size):
        self.input = input

        self.W = self._init((in_size, out_size * 4))
        self.U = self._init((in_size, out_size * 4))
        self.b = np.zeros(out_size, dtype=theano.config.floatX)

        def slice(x, n, dim):
            return x[n * dim:(n + 1) * dim]

        def step(x, h_, c_):
            preact = tt.dot(h_, self.U)
            preact += x

            i = tt.nnet.sigmoid(slice(preact, 0, out_size))
            f = tt.nnet.sigmoid(slice(preact, 1, out_size))
            o = tt.nnet.sigmoid(slice(preact, 2, out_size))
            c = tt.tanh(slice(preact, 3, out_size))

            c = f * c_ + i * c
            h = o * tt.tanh(c)

            return h, c

        tmp = tt.dot(self.input, self.W) + self.b

        vals, _ = theano.scan(step, sequences=[tmp], outputs_info=[
            tt.alloc(np.asarray(0., dtype=theano.config.floatX), out_size),
            tt.alloc(np.asarray(0., dtype=theano.config.floatX), out_size)
        ])

        self.output = vals[0]
        self.params = [self.W, self.U, self.b]

    def _init(self, shape):
        return np.asarray(
            RNG.uniform(low=-1, high=1, size=shape),
            dtype=theano.config.floatX
        )


class CTCLayer(object):

    def __init__(self, input, y):
        self.input = input

        def recurrence_relation(size):
            big_I = tt.eye(size+2)
            return (
                tt.eye(size) + big_I[2:,1:-1] + big_I[2:,:-2] *
                (tt.arange(size) % 2)
            )

        P = tt.nnet.softmax(self.input)[:, y]
        rr = recurrence_relation(y.shape[0])

        def step(curr, prev):
            return curr * tt.dot(prev,rr)

        probs,_ = theano.scan(
            step,
            sequences = [P],
            outputs_info = [T.eye(Y.shape[0])[0]]
        )

        self.output = probs
