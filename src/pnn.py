import torch
from torch import nn
import numpy as np

basex = torch.tensor([1, 0, 0], dtype = float, requires_grad=False)
basey = torch.tensor([0, 1, 0], dtype = float, requires_grad=False)
basez = torch.tensor([0, 0, 1], dtype = float, requires_grad=False)

def objectTensor(r, v = (0, 0, 0), a = (0, 0, 0), 
                 size = 0):
    """An object is represented by a tensor
    shape: [B, N] or [N]
    B: batch
    N = 10, [r1, r2, r3, v1, v2, v3, a1, a2, a3, size]
    """
    t = list(r[0:3]) + list(v[0:3]) + list(a[0:3]) + [size]
    return torch.tensor(t)

def decodeObject(t : torch.tensor):
    """Decompose 1 object tensor to r, v, a
    """
    if len(t.shape) == 1:
        r, v, a, sz = t[:3], t[3:6], t[6:9], t[9]
    else:
        # batch
        r, v, a, sz = t[:, :3], t[:, 3:6], t[:, 6:9], t[:, 9]
    return r, v, a, sz

class MotionNN:
    """Motion of objects
    """
    @staticmethod
    def moved(t: torch.tensor, dt : float) -> torch.tensor:
        """Assume dt is small
        Change the state of this object tensor
        """
        s = t.clone()
        s[...,0:3] = t[..., 0:3] + t[..., 3:6] * dt
        s[...,3:6] = t[..., 3:6] + t[..., 6:9] * dt
        return s

    @staticmethod
    def reflected(t: torch.tensor, ground : float = -1.5) -> None:
        """reflected by the ground
        """
        for i in range(t.shape[0]):
            if t[i, 2] <= ground:
                t[i, 2] = ground
                if t[i, 5] < 0:
                    t[i, 5] = - t[i, 5]

    @staticmethod
    def deflected(t: torch.tensor, impulse: torch.tensor) -> None:
        """Sudden change of v
        """
        t[..., 3:6] += impulse
        
    
    @staticmethod
    def acc(t: torch.tensor, slow : float = 0, curve: float = 0,
            gravity = 10) -> None:
        """slowdonw the velocity by a factor 
           and centrapital acceleartion
        """
        t[..., 6:9] =  - basez * gravity
        if slow > 0 and slow < 1:
            t[..., 6:9] += t[..., 3:6] * (slow - 1)
        
        if curve>0:
            a = torch.cross(t[..., 3:6].double(), basez.unsqueeze(0),
                        dim = -1)
            t[..., 6:9] += a * curve


    @staticmethod
    def freeFall(t: torch.tensor) -> None:
        """Change the accelerator to free fall
        """
        t[..., 6] = 0
        t[..., 7] = 0
        t[..., 8] = 10

    @staticmethod
    def constVec(t: torch.tensor) -> None:
        """set a to be 0
        """
        t[..., 6:9] = 0

    @staticmethod
    def idle(t: torch.tensor) -> None:
        """set v and a be 0
        """ 
        t[..., 3:9] = 0

        

def object2box(t: torch.tensor, sz = 0.1):
    """Convert this tensor to center and bbox
    The last dimension [N = 10] -> [9, 3]
    where 9 corresponds to 9 points, center and vertexes
    3 corresponds to their coordinates
    """
    a = torch.zeros([*t.shape[:-1], 9, 3], dtype = float)
    # sz = t[...,9] / 2
    a[..., 0, :] = t[..., 0:3]
    a[..., 1, :] = t[..., 0:3] + (-basex - basey - basez) * sz
    a[..., 2, :] = t[..., 0:3] + (-basex + basey - basez) * sz
    a[..., 3, :] = t[..., 0:3] + (basex + basey - basez) * sz
    a[..., 4, :] = t[..., 0:3] + (basex - basey - basez) * sz
    sz *= 2
    a[..., 5, :] = a[..., 1, :] + basez * sz
    a[..., 6, :] = a[..., 2, :] + basez * sz
    a[..., 7, :] = a[..., 3, :] + basez * sz
    a[..., 8, :] =a [..., 4, :] + basez * sz
    return a


class ProjectorNN(nn.Module):
    def __init__(self, basis = None):
        super(ProjectorNN, self).__init__()
        # Initialize a learnable 3x3 tensor
        if basis is None:
            self.basis = nn.Parameter(torch.randn(3, 3, dtype = float))
        else:
            self.basis = torch.tensor(basis, requires_grad=True, dtype = float)
        
        self.zoom = 1000 * 0.2  # hs * zoom

    def forward(self, x):
        """
        Forward pass that performs matrix multiplication.
        Input:
            x - A tensor of shape (3,) or (batch_size, 3)
        Output:
            A tensor of shape (3,) or (batch_size, 3), result of matrix multiplication.
        """
        x = torch.matmul(self.basis, x.mT).mT  # Handles both single and batched input

        """
        Divide all entries by the first entry along the specified dimension, except for the batch dimension.
        :param x: Input tensor of shape (batch_size, ...).
        :return: Tensor with entries divided by the first element along the specified dimension.
        """
        dim = -1
        if x.size(dim) == 0:
            raise ValueError("Specified dimension is empty.")
        
        # Select the first entry along the specified dimension for each batch
        first_entry = x.select(dim, 0).unsqueeze(dim)
        
        # Ensure no division by zero
        if torch.any(first_entry == 0):
            raise ValueError("Division by zero detected in the first entry of the specified dimension.")
        
        y = x / first_entry * self.zoom
        y = y[..., 1:]   # removing the 1st entry

        """Keep the bbox, [min, max], [..., n, 2] -> [..., 2, 2]
        """
        y1, _ = torch.min(y, dim = -2)
        y2, _ = torch.max(y, dim = -2)
        # print("y, y1, y2 =", y.shape, y1.shape, y2.shape)
        y = torch.stack((y1, y2), dim = -1)
        return y
