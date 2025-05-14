import torch

class AverageMeter(object):
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items
        if self.items is None:
            self._values = []  # store values for single item
        else:
            self._values = [[] for _ in range(self.n_items)]  # store values for each item

    def update(self, values):
        if type(values).__name__ == 'list':
            for idx, v in enumerate(values):
                self._val[idx] = v
                self._sum[idx] += v
                self._count[idx] += 1
                self._values[idx].append(v)

        else:
            self._val[0] = values
            self._sum[0] += values
            self._count[0] += 1
            self._values.append(values)

    def val(self, idx=None):
        if idx is None:
            return self._val[0] if self.items is None else [self._val[i] for i in range(self.n_items)]
        else:
            return self._val[idx]

    def count(self, idx=None):
        if idx is None:
            return self._count[0] if self.items is None else [self._count[i] for i in range(self.n_items)]
        else:
            return self._count[idx]

    def avg(self, idx=None):
        if idx is None:
            return self._sum[0] / self._count[0] if self.items is None else [
                self._sum[i] / self._count[i] for i in range(self.n_items)
            ]
        else:
            return self._sum[idx] / self._count[idx]
        
    def median(self, idx=None):
        if self.items is None:
            if not self._values:
                return None
            tensor_vals = torch.tensor(self._values)
            return torch.median(tensor_vals).item()
        else:
            if idx is None:
                medians = []
                for vals in self._values:
                    if vals:
                        tensor_vals = torch.tensor(vals)
                        medians.append(torch.median(tensor_vals).item())
                    else:
                        medians.append(None)
                return medians
            else:
                vals = self._values[idx]
                if not vals:
                    return None
                tensor_vals = torch.tensor(vals)
                return torch.median(tensor_vals).item()