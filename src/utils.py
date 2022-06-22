"""
This file contains logic for loading the specific optimizer
and criterion for a specific class.
"""
import os

# TODO
def get_optim():
    raise NotImplementedError


# TODO
def get_criterion():
    raise NotImplementedError


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()
        self.end = time.time()

    def elapsed(self):
        self.end = time.time()
        return self.end - self.start

    def minutes_elapsed(self):
        return self.elapsed() / 60


class AverageMeter:
    """This function tracks losses of the model.
    Code taken from class AverageMeter()
        https://github.com/facebookresearch/swav/blob/main/src/utils.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PerformanceMonitor:
    def __init__(self, target_dir):
        self.target_dir = os.path.join(target_dir, "performance.csv")
        self.columns = [
            "epoch",
            "stage",
            "loss",
        ]

        with open(self.target_dir, "w") as f:
            f.write(",".join(self.columns))
            f.write("\n")

    def log(self, stage, epoch, scores, minutes, kld_weight=None):
        """open the file and append a line to the csv"""
        with open(self.target_dir, "a") as f:
            f.write(",".join([str(epoch), stage, str(loss),]))
            f.write("\n")
