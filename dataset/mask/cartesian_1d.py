import numpy as np

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

class MaskFunc:
    def __init__(self, cf, acc, same=False):
        self.cf = cf
        self.acc = acc
        self.same = same
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None, centred=False):
        assert len(shape) == 3

        self.rng.seed(seed)
        N = shape[0]

        size = shape[-1]
        sample_n = int(self.cf*size)
        pdf_x = normal_pdf(size, 0.5/(size/10.)**2)
        lmda = size/(2.*self.acc)
        n_lines = int(size / self.acc)

        # add uniform distribution
        pdf_x += lmda * 1./size

        if sample_n:
            pdf_x[size//2-sample_n//2:size//2+sample_n//2] = 0
            pdf_x /= np.sum(pdf_x)
            n_lines -= sample_n

        mask = np.zeros((N, size))
        if self.same:
            idx = self.rng.choice(size, n_lines, False, pdf_x)
            mask[:, idx] = 1
        else:
            for i in range(N):
                idx = self.rng.choice(size, n_lines, False, pdf_x)
                mask[i, idx] = 1

        if sample_n:
            mask[:, size//2-sample_n//2:size//2+sample_n//2] = 1
            pdf_x = np.sum(mask, axis=0, keepdims=True)
            pdf_x = pdf_x/pdf_x.max()

        if not centred:
            mask = np.fft.ifftshift(mask, axes=(-1, -2))
        mask = np.expand_dims(mask, 1)
        return [mask, pdf_x]
