class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_epsilon(self, current_step):
        return self.end + (self.start - self.end) * np.exp(-1. * current_step / self.decay)
