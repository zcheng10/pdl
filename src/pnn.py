import torch
from torch import nn
import torch.optim as optim

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
    return torch.tensor(t, dtype = float)

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
    def moved(t: torch.tensor, N:int = 1, dt:float = 0.03) ->torch.tensor:
        """Move a state for N times, output the corresponding
            states, [N, 10]
        """
        s = torch.zeros([N, 10])
        A = torch.zeros([N, 10, 10])
        A.requires_grad = False





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
    sz /= 2
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
        y = torch.stack((y1, y2), dim = -1).transpose(-1, -2)
        return y
    

class ActionNN (nn.Module):
    """Moving an object for N times and output the N
        states [N, 15]
    """
    def __init__(self, N: int, dt:float = 0.03) -> None:
        super(ActionNN, self).__init__()
        layers = []
        for _ in range(N):
            layers.append(MotionNN.moved)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        """


class MotionSolver:
    """Given a sequence of bbox, find (r, v, a), and predict 
    the next location
    """
    def __init__(self):
        self.dt = 0.03
            
        self.pj = ProjectorNN(
            basis = [[1, 0, 0], [0, 1, 0], 
                     [0, 0, 1]])
        self.pj.basis.requires_grad = False
        self.pj.zoom = 1000
        

    def solve(self, input: torch.tensor, 
              guess: torch.tensor = None,
              predictions: int = 4,
              epochs: int = 100,
              err_tol = 1.0,
              verbose: bool = False):
        """Solve for pose
        Args:
            input: the input tensors [B, N], N = 4
                N = 4 corresponds to bbox in each frame
                (normalized to (-1, 1))
                B is the number of input frames

            guess: the starting value of (r, v, a, sz)
            predictions: the number of predictions to make
            epochs: the number of epochs for the solution
            err_tol: when loss < err_rol, stop solving
            
        Returns:
            pose: the estimated pose, [10]
            next_bbox: the next predicted bboxes, [P, 2, 2]
            next_poses: the next predicted states, [P, 10]
        """
        if guess is None:
            pose = objectTensor(r = (10, 0, -1.5),
                                size = 0.22)
        else:
            pose = guess
        
        pose.requires_grad = True

        # get the transition matrix
        N = input.shape[0]
        T = MotionSolver.transition(N, self.dt)

        # Define the optimizer
        # Use Adam optimizer with learning rate 0.1
        optimizer = optim.Adam([pose], lr=0.1)
        tgt = input.flatten().double()
        if verbose:
            print("tgt =", tgt.detach().numpy())
        
        # Define MSE loss function
        mse_loss = nn.MSELoss()

        # Optimization loop
        for step in range(epochs):  # Number of iterations
            optimizer.zero_grad()  # Clear gradients from the previous step
            
            # forward pass
            x = torch.matmul(T, pose)
            x = object2box(x, sz = 0.22)
            y = self.pj(x)
            y = y.flatten()

            # Calculate loss
            loss = mse_loss(y, tgt)

            if loss.item() < err_tol:
                break;

            # Compute gradients
            loss.backward()
                
            # Update the parameter
            optimizer.step()
                
            # Print progress
            if verbose:
                if step % 10 == 0:  # Print every 10 steps
                    print(f"Step {step}: x = {pose.detach().numpy()}")
                    print(f"    loss = {loss.item()}, y = {y.detach().numpy()}")

        # Final result
        if verbose:
            print(f"Optimized x = {pose.detach().numpy()}, Minimum value = {loss.item()}")

        # prediction
        x = torch.matmul(T, pose)
        x = x[-1, :]    # last state
        A = MotionSolver.transition(predictions + 1, dt = self.dt)
        y = torch.matmul(A, x)
        next_poses = y.clone()[1,...]
        y = object2box(y, sz = 0.22)
        y = self.pj(y)
        next_states = y[1:, ...]    # last epochs

        return pose, next_states, next_poses

    
    @staticmethod
    def transition(N:int = 1, dt: float = 0.03):
        """Compute the transition matrix, so N states can be 
        computed. The input tensor shape is [10]. The output tensor
        shape is [N, 10]. So the transition matrix is [N, 10, 10].

        Position: t[0:3] -> t[0:3] + t[3:6] * dt + 0.5*t[6:9]* dt^2
        Velocity: t[3:6] -> t[3:6] + t[6:9] * dt
        """
        batch_size = N
        n = 10
        A = torch.zeros([N, n, n], dtype = float)
        A.requires_grad = False

        for i in range(N):
            x, y = i * dt, 0.5 * (i * dt) **2
            A[i, :, :] = torch.eye(n)
            for j in range(0, 3):
                A[i, j, j + 3] = x
                A[i, j, j + 6] = y
            for j in range(3, 6):
                A[i, j, j + 3] = x
        return A
