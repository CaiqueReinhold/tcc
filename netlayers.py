import numpy as np
import theano
import theano.tensor as tt
import theano.tensor.nnet.conv as conv
from theano.tensor.signal.pool import pool_2d


RNG = np.random.RandomState(1234)


class DenseLayer(object):

    def __init__(self, input, in_size, out_size, params=None,
                 activation=tt.tanh):
        self.input = input

        if params is None:
            W_values = self._init((in_size, out_size))
            b_values = np.zeros((out_size,), dtype=theano.config.floatX)
        else:
            W_values = params[0]
            b_values = params[1]

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

    def __init__(self, filter_shape, image_shape, params=None,
                 poolsize=(2, 2), activation=tt.tanh):
        assert image_shape[1] == filter_shape[1]

        if params is None:
            W_values = self._init(filter_shape)
            b_values = np.zeros((filter_shape[0],),
                                dtype=theano.config.floatX)
        else:
            W_values = params[0]
            b_values = params[1]

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation = activation

        self.params = [self.W, self.b]

    def output(self, input):
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape
        )

        pooled_out = pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True
        )

        return self.activation(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        )

    def _init(self, shape):
        fan_in = np.prod(shape[1:])
        fan_out = (shape[0] * np.prod(shape[2:]) / 4)

        W_bound = np.sqrt(6. / (fan_in + fan_out))
        return np.asarray(
            RNG.uniform(low=-W_bound, high=W_bound, size=shape),
            dtype=theano.config.floatX
        )


class LSTMLayer(object):

    def __init__(self, input, in_size, out_size, params=None):
        self.input = input

        if params is None:
            W_values = self._init((in_size, out_size))
            U_values = self._init((out_size, out_size))
            b_values = np.zeros(out_size * 4, dtype=theano.config.floatX)
        else:
            W_values = params[0]
            U_values = params[1]
            b_values = params[2]

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.U = theano.shared(value=U_values, name='U', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

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
            np.zeros((out_size,), dtype=theano.config.floatX),
            np.zeros((out_size,), dtype=theano.config.floatX),
        ])

        self.output = vals[0]
        self.params = [self.W, self.U, self.b]

    def _init(self, shape):
        return np.hstack([
            self._ortho_weight(shape[0], shape[1]),
            self._ortho_weight(shape[0], shape[1]),
            self._ortho_weight(shape[0], shape[1]),
            self._ortho_weight(shape[0], shape[1])
        ])

    def _ortho_weight(self, r, c):
        W = np.random.randn(r, c)
        u = np.linalg.svd(W)[0]
        return u[:r, :c].astype(theano.config.floatX)


class CTCLayer(object):

    def __init__(self, input, y):
        import prepare_data
        self.input = tt.nnet.softmax(input)
        self.y = y
        self.blank = len(prepare_data.CLASSES)
        self.output = tt.argmax(self.input, axis=1)

    def plain_ctc(self):
        l = tt.concatenate((self.y, [self.blank, self.blank]))
        sec_diag = tt.neq(l[:-2], l[2:]) * tt.eq(l[1:-1], self.blank)

        recurrence_relation = (
            tt.eye(self.y.shape[0]) +
            tt.eye(self.y.shape[0], k=1) +
            tt.eye(self.y.shape[0], k=2) *
            sec_diag.dimshuffle((0, 'x'))
        )

        pred_y = self.input[:, self.y]

        probs, _ = theano.scan(
            lambda curr, accum: curr * tt.dot(accum, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[tt.eye(self.y.shape[0])[0]]
        )

        l_probs = tt.sum(probs[-1, -2:])
        return -tt.log(l_probs)

    def log_ctc(self):
        def safe_log(x):
            return tt.log(tt.maximum(x, 1e-20).astype(theano.config.floatX))

        def logmul(x, y):
            return x + y

        def safe_exp(x):
            return tt.exp(tt.minimum(x, 1e20).astype(theano.config.floatX))

        def logadd_simple(x, y):
            return x + safe_log(1 + safe_exp(y - x))

        def logadd(x, y, *zs):
            sum = logadd_simple(x, y)
            for z in zs:
                sum = logadd_simple(sum, z)
            return sum

        prev_mask = 1 - tt.eye(self.y.shape[0])[0]
        prevprev_mask = (
            tt.neq(self.y[:-2], self.y[2:]) *
            tt.eq(self.y[1:-1], self.blank)
        )
        prevprev_mask = tt.concatenate(([0, 0], prevprev_mask))
        prev_mask = safe_log(prev_mask)
        prevprev_mask = safe_log(prevprev_mask)
        prev = tt.arange(-1, self.y.shape[0]-1)
        prevprev = tt.arange(-2, self.y.shape[0]-2)
        log_pred_y = tt.log(self.input[:, self.y])

        def step(curr, accum):
            return logmul(
                curr,
                logadd(
                    accum,
                    logmul(prev_mask, accum[prev]),
                    logmul(prevprev_mask, accum[prevprev])
                )
            )

        log_probs, _ = theano.scan(
            step,
            sequences=[log_pred_y],
            outputs_info=[safe_log(tt.eye(self.y.shape[0])[0])]
        )

        return -log_probs[-1, -1]
