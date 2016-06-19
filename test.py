import theano
import theano.tensor as tt
import numpy as np

import netlayers
import prepare_data


tensor_x = theano.tensor.matrix('x')
tensor_y = theano.tensor.ivector('y')

lstm = netlayers.LSTMLayer(tensor_x, len(prepare_data.CLASSES), len(prepare_data.CLASSES))
out = netlayers.DenseLayer(lstm.output, len(prepare_data.CLASSES), len(prepare_data.CLASSES)+1)
ctc = netlayers.CTCLayer(out.output, tensor_y)

_, y = prepare_data.get_data()

def y_to_x(y):
    n = np.zeros((93, len(prepare_data.CLASSES)))
    for i in range(len(y)):
        n[i][y[i]] = 1.0
    return n.astype(theano.config.floatX)

x = [y_to_x(a) for a in y]

p_y_given_x = tt.nnet.softmax(out.output)
y_pred = tt.argmax(p_y_given_x, axis=1)

pad_y = tt.concatenate([tensor_y, tt.zeros((93 - tensor_y.shape[0],), dtype='int64')])

negative_log_likehood = -tt.mean(tt.log(p_y_given_x)[tt.arange(pad_y.shape[0]), pad_y])

errors = tt.mean(tt.neq(y_pred, pad_y))

cost = ctc.log_ctc()

learning_rate = 0.001
grads = tt.grad(cost, lstm.params + out.params)
updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(lstm.params, grads)
]
train_model = theano.function(
    inputs=[tensor_x, tensor_y],
    outputs=cost,
    updates=updates
)
test = theano.function(
    inputs=[tensor_x, tensor_y],
    outputs=errors
)

for epoch in range(100):
    mean_costs = [train_model(x[i], y[i]) for i in range(len(x))]
    mean_errors = [test(x[j], y[j]) for j in range(len(x))]
    print 'epoch', epoch, 'mean error:' , np.mean(mean_errors), 'mean cost:', np.mean(mean_costs)