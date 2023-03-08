from jittor import nn
import jittor as jt
import numpy as np

class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        faces = faces.int().numpy()
        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= laplacian[i, i]

        self.laplacian = jt.array(laplacian).stop_grad()

    def execute(self, x):
        batch_size = x.size(0)
        x = jt.matmul(self.laplacian, x)
        dims = tuple(range(len(x.shape))[1:])
        x = x.pow(2).sum(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x