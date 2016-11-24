# Implementing Recurrent Neural Network from Scratch
I’m assuming that you are somewhat familiar with basic Neural Networks. If you’re not, you may want to head over to [Implementing A Neural Network From Scratch](https://github.com/pangolulu/neural-network-from-scratch),  which guides you through the ideas and implementation behind non-recurrent networks.
## Introduction
This post is inspired by [recurrent-neural-networks-tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) from [WildML](http://www.wildml.com/). And you can deeply read it to know the basic knowledge about RNN, which I will not include in this tutorial.

In this tutorial, we will focus on how to train RNN by [Backpropagation Through Time (BPTT)](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/), based on the **computation graph** of RNN and do **automatic differentiation**. You can find that it is more simple and reliable to calculate the gradient in this way than you do it by hand.

This post will take RNN language model (rnnlm) as example. More about the fancy applications of RNN can be found [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
## How to train RNN
The architecture of RNN can be as the following figure.

![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn.jpg)

You can find that the parameters `(W, U, V)` are shared in different time steps. And the output in each time step can be **softmax**. So you can use **cross entropy** loss as an error function and use some optimizing method (e.g. gradient descent) to calculate the optimized parameters `(W, U, V)`.

Let recap the equations of our RNN:

![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn_equation.png)

We also defined our loss, or error, to be the cross entropy loss, given by:

![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn_loss.png)

Here `y_t` is the correct word at time step `t`, and `y^_t` is our prediction. We typically treat the full sequence (sentence) as one training example, so the total error is just the sum of the errors at each time step (word).

![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn-bptt1.png)

Remember that our goal is to calculate the gradients of the error with respect to our parameters `U`, `V` and `W` and then learn good parameters using optimizing method (in this post we use **Stochastic Gradient Descent**). Just like we sum up the errors, we also sum up the gradients at each time step for one training example:  ![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/gradient.png). That is we should calculate `dEt/dW`, `dEt/dU` and `dEt/dV`, then sum up all time steps.

It is simple to calculate `dEt/dV`, because it only depends on the values at the current time step. But the story is different for `dEt/dW` and `dEt/dU`. Note that `s_3 = tanh(Ux_3 + Ws_2)` depend on `s_2`, which depends on `W`, `U` and `s_1`, and so on.  So if we take the derivative with respect to `W` we can't treat `s_2` as a constant! We need to apply the chain rule again. You can have a view from the following figure.

![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn-bptt-with-gradients.png)

Now use **computation graph** to represent `E1` as an example and calculate `dE1/dW`, `dE1/dU` is the same idea.

![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn-compuattion-graph.png)

Note that this is exactly the same as the standard backpropagation algorithm that we use in deep [Feedforward Neural Networks](https://github.com/pangolulu/neural-network-from-scratch). The key difference is that we sum up the gradients for `W` at each time step. In a traditional NN we don’t share parameters across layers, so we don’t need to sum anything.  But in my opinion BPTT is just a fancy name for standard backpropagation on an unrolled RNN.

To simplify the **computation graph** to make it efficient, we can integrate some small operation units to a big operation unit. You can have a look the following figure. Note that the operation unit should also implement the `forward` function and `backward` function.

![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn-compuattion-graph_2.png)

The implementation of all operation unit and softmax output can be found as follows:
```python
mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()

class RNNLayer:
    def forward(self, x, prev_s, U, W, V):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)
        
    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV)
```

```python
class MultiplyGate:
    def forward(self,W, x):
        return np.dot(W, x)
    def backward(self, W, x, dz):
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        dx = np.dot(np.transpose(W), dz)
        return dW, dx

class AddGate:
    def forward(self, x1, x2):
        return x1 + x2
    def backward(self, x1, x2, dz):
        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)
        return dx1, dx2
```

```python
class Sigmoid:
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - output) * output * top_diff

class Tanh:
    def forward(self, x):
        return np.tanh(x)
    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - np.square(output)) * top_diff
```

```python
class Softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)
    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[y])
    def diff(self, x, y):
        probs = self.predict(x)
        probs[y] -= 1.0
        return probs
```

These implementation is just the same with [Implementing A Neural Network From Scratch](https://github.com/pangolulu/neural-network-from-scratch), except that in this post the input `x` or `s` is `1-D array`, but in previous post input `X` is a batch of data represented as a matrix (each row is an example).

Now that we are able to calculate the gradients for our parameters we can use SGD to train the model.

## Implement
### Initialization
Initializing the parameters  `U`, `V` and `W` is a bit tricky. We can’t just initialize them to 0’s because that would result in symmetric calculations in all our layers. We must initialize them randomly. Because proper initialization seems to have an impact on training results there has been lot of research in this area. It turns out that the best initialization depends on the activation function (`tanh` in our case) and one recommended approach is to initialize the weights randomly in the interval from ![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/init.png) where `n` is the number of incoming connections from the previous layer. 
```python
class Model:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
```

Above, `word_dim` is the size of our vocabulary, and `hidden_dim` is the size of our hidden layer (we can pick it). Don’t worry about the `bptt_truncate` parameter for now, we’ll explain what that is later.
### Forward Propagation
Next, let’s implement the forward propagation (predicting word probabilities) defined by our equations above:
```python
'''
    forward propagation (predicting word probabilities)
    x is one single data, and a batch of data
    for example x = [0, 179, 341, 416], then its y = [179, 341, 416, 1]
'''
def forward_propagation(self, x):
    # The total number of time steps
    T = len(x)
    layers = []
    prev_s = np.zeros(self.hidden_dim)
    # For each time step...
    for t in range(T):
        layer = RNNLayer()
        input = np.zeros(self.word_dim)
        input[x[t]] = 1
        layer.forward(input, prev_s, self.U, self.W, self.V)
        prev_s = layer.s
        layers.append(layer)
    return layers
```

We also implement a `predict` function to generate the results.
```python
def predict(self, x):
    output = Softmax()
    layers = self.forward_propagation(x)
    return [np.argmax(output.predict(layer.mulv)) for layer in layers]
```

### Calculating the Loss
To train our network we need a way to measure the errors it makes. We call this the loss function `L`, and our goal is find the parameters `U`, `V` and `W` that minimize the loss function for our training data. A common choice for the loss function is the **cross-entropy** loss. 
```python
def calculate_loss(self, x, y):
    assert len(x) == len(y)
    output = Softmax()
    layers = self.forward_propagation(x)
    loss = 0.0
    for i, layer in enumerate(layers):
        loss += output.loss(layer.mulv, y[i])
    return loss / float(len(y))

def calculate_total_loss(self, X, Y):
    loss = 0.0
    for i in range(len(Y)):
        loss += self.calculate_loss(X[i], Y[i])
    return loss / float(len(Y))
```

### Backpropagation Through Time (BPTT)
Just as what we have introduced, we implement BPTT algorithm.  It takes as input a training example `(x, y)` and returns the gradients `dL/dW`, `dL/dU` and `dL/dV`.
```python
def bptt(self, x, y):
    assert len(x) == len(y)
    output = Softmax()
    layers = self.forward_propagation(x)
    dU = np.zeros(self.U.shape)
    dV = np.zeros(self.V.shape)
    dW = np.zeros(self.W.shape)

    T = len(layers)
    prev_s_t = np.zeros(self.hidden_dim)
    diff_s = np.zeros(self.hidden_dim)
    for t in range(0, T):
        dmulv = output.diff(layers[t].mulv, y[t])
        input = np.zeros(self.word_dim)
        input[x[t]] = 1
        dprev_s, dU_t, dW_t, dV_t = layers[t].backward(input, prev_s_t, self.U, self.W, self.V, diff_s, dmulv)
        prev_s_t = layers[t].s
        dmulv = np.zeros(self.word_dim)
        for i in range(t-1, max(-1, t-self.bptt_truncate-1), -1):
            input = np.zeros(self.word_dim)
            input[x[i]] = 1
            prev_s_i = np.zeros(self.hidden_dim) if i == 0 else layers[i-1].s
            dprev_s, dU_i, dW_i, dV_i = layers[i].backward(input, prev_s_i, self.U, self.W, self.V, dprev_s, dmulv)
            dU_t += dU_i
            dW_t += dW_i
        dV += dV_t
        dU += dU_t
        dW += dW_t
    return (dU, dW, dV)
```

### SGD Implementation
Now that we are able to calculate the gradients for our parameters we can implement SGD. I like to do this in two steps:

1. A function `sdg_step` that calculates the gradients and performs the updates for one batch.
2. An outer loop that iterates through the training set and adjusts the learning rate.

```python
def sgd_step(self, x, y, learning_rate):
    dU, dW, dV = self.bptt(x, y)
    self.U -= learning_rate * dU
    self.V -= learning_rate * dV
    self.W -= learning_rate * dW
   
def train(self, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    num_examples_seen = 0
    losses = []
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = self.calculate_total_loss(X, Y)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(Y)):
            self.sgd_step(X[i], Y[i], learning_rate)
            num_examples_seen += 1
    return losses 
```

Here, we annealing the learning rate by `0.5` if we find the loss increases in this epoch. More about the decay of learning rate can be found [here](http://cs231n.github.io/neural-networks-3/#anneal).
## Evaluation
Done! Let’s try to get a sense of how long it would take to train our network:
```python
word_dim = 8000
hidden_dim = 100
X_train, y_train = getSentenceData('data/reddit-comments-2015-08.csv', word_dim)

np.random.seed(10)
rnn = Model(word_dim, hidden_dim)
rnn.sgd_step(X_train[10], y_train[10], 0.005)
```

Bad new is that  one step of SGD takes a few seconds on my laptop. We have about 80,000 examples in our training data, so one epoch (iteration over the whole data set) would take several hours. Multiple epochs would take days, or even weeks! 

There are many ways to speed up our code. One is to implement our code on GPU with some library like **Theano**. But in this tutorial, let’s just try to run SGD with a small dataset and check if the loss actually decreases:
```python
word_dim = 8000
hidden_dim = 100
X_train, y_train = getSentenceData('data/reddit-comments-2015-08.csv', word_dim)

np.random.seed(10)
rnn = Model(word_dim, hidden_dim)

losses = rnn.train(X_train[:100], y_train[:100], learning_rate=0.005, nepoch=10, evaluate_loss_after=1)
```

![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn_eval.png)

Good, it seems like our implementation is at least doing something useful and decreasing the loss, just like we wanted.
## Further more
There is a problem about RNN called `vanishing gradient problem`. That's why traditional RNN cannot capture the long term dependency, so we use `bptt_truncate` parameter to constrain the length of dependency. This will motivate our move to more sophisticated RNN models, such as LSTMs, which are the current state of the art for many tasks in NLP. 

More about `vanishing gradient problem` and `LSTM` can be found [here](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/) and [here](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/) and [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).
## Reference
1. <https://github.com/pangolulu/neural-network-from-scratch>
2. <http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/>
2. <http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/>
3. <http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/>
