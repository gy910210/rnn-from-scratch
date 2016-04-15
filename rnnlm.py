import numpy as np

from preprocessing import getSentenceData
from rnn import Model

word_dim = 8000
hidden_dim = 100
X_train, y_train = getSentenceData('data/reddit-comments-2015-08.csv', word_dim)

np.random.seed(10)
rnn = Model(word_dim, hidden_dim)

losses = rnn.train(X_train[:100], y_train[:100], learning_rate=0.005, nepoch=10, evaluate_loss_after=1)