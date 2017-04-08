class PerformanceParams:
    def __init__(self):
        self.true = 0.
        self.false = 0.
        self.accuracy = 0.



    def get_accuracy(self):
        self.accuracy = (self.true ) / (self.true + self.false)
        self.accuracy = self.accuracy*100