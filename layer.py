from activation import Tanh
from gate import AddGate, MultiplyGate

mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()

class RNNLayer:
    def forward(self, x, previous_s, U, W, V):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, previous_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, previous_s)

    def backward(self, x, previous_s, U, W, V, diff_s, dmulv):
        self.forward(x, previous_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprevious_s = mulGate.backward(W, previous_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprevious_s, dU, dW, dV)