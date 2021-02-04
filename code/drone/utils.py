from multiprocessing import Pool


class mzip():

    def __init__(self, *args):
        self.iters = list(map(iter, args))

    def __iter__(self):
        with Pool() as pool:
            while True:
                yield tuple(pool.map(next, self.iters))
