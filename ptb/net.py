import chainer
import chainer.functions as F
import chainer.links as L


class RNNLM(chainer.Chain):

    """Recurrent neural net languabe model for penn tree bank corpus.

    This is an example of deep LSTM network for infinite length input.

    """
    def __init__(self, n_vocab, n_units, train=True):
        super(RNNLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        y = self.l3(F.dropout(h2, train=self.train))
        return y


class PNLM(chainer.Chain):

    def __init__(self, n_in, n_units, train=True):
        super(PNLM, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.LSTM(n_units, n_units),
            l4=L.Linear(n_units, n_in),
        )
        self.train = train

    def reset_state(self):
        self.l2.reset_state()
        self.l3.reset_state()

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(F.dropout(h1, train=self.train))
        h3 = self.l3(F.dropout(h2, train=self.train))
        y = self.l4(F.dropout(h3, train=self.train))
        return y


if __name__ == '__main__':
    filename = 'hoge'
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    with open("new"+filename, 'w') as f:
        for word in words:
            f.write(word+" ")
