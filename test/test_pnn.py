import unittest
import sys

sys.path.append("../")
from src.pworld import *
from src.pnn import *
from src.illustrate import *

class TestBasic(unittest.TestCase):

    BALLS = [
        [10, 0, 2],
        [10, 0, 10],
        [10, 8, 10]
    ]

    BALL_SIZE = 0.1

    BASIS = arr([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype = float)

    def test_Object(self):
        """Test Object"""
        obs = [Object("ball1", size = TestBasic.BALL_SIZE, 
                   r = arr(TestBasic.BALLS[0]) ),
            Object("ball2", size = TestBasic.BALL_SIZE, 
                   r = arr(TestBasic.BALLS[1])),
            Object("ball3", size = TestBasic.BALL_SIZE, 
                   r = arr(TestBasic.BALLS[2]))]
        
        p = Projector(hs = 0.2, h0 = TestBasic.BASIS[0], 
                      h1 = TestBasic.BASIS[1])
        print("p =", p)

        br = [p.toScreen(a) for a in obs]
            
        for i, a in enumerate(obs):
            print("a =", a)
            print("br =", br[i])
            print("--------")

        # self.assertEqual(square(3), 9)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        P3.box(arr([0,0,0]), size=0.05, 
           color = "green", alpha = 1, ax = ax)            
        
        for i, a in enumerate(obs):
            P3.box(a.r, size = a.size, color = "red", alpha = 1, ax = ax)

        P3.rectangle_3d(-p.h0 * p.hs - p.h1/2 - p.h2/2, 
                        p.h0 * 0, p.h1, p.h2, ax = ax)

        ax.axis("equal")
       
    def test_projectNN(self):
        module = ProjectorNN(basis = TestBasic.BASIS)
        # Single input tensor (vector of size 3)
        x_single = torch.tensor([TestBasic.BALLS[0]], requires_grad=True,
                                dtype = float)
        
        # Forward pass for single tensor
        output_single = module(x_single)
        print("Output (single input):", output_single)

        # Batched input tensor (batch of vectors, shape [batch_size, 3])
        x_batch = torch.tensor(TestBasic.BALLS[:2], requires_grad=True,
                               dtype=float)
        
        # Forward pass for batch input
        output_batch = module(x_batch)
        print("Output (batch input):", output_batch)


    def test_objectTensor(self):
        t = objectTensor(r = TestBasic.BALLS[0], a = (0, 0, 10), 
                         size = 0.1).double()
        t.requires_grad = True
        print("object tensor =", t)

        r, v, a, sz = decodeObject(t)
        print("decoded to", r, v, a, sz)

        a = object2box(t)
        # a.requires_grad = False
        print("bbox =", a)

        m = ProjectorNN(basis = TestBasic.BASIS)
        m.basis.requires_grad = False
        y = m(a)
        print("projected box =", y)

        y.sum().backward()
        print("object grad =", t.grad)
        print("projector grad =", m.basis.grad)


    def test_objectMoving(self):
        t = objectTensor(r = TestBasic.BALLS[0], a = (0, 0, 10), 
                         size = 0.1).double()
        t.requires_grad = True
        s = MotionNN.moved(t, dt = 0.1)
        print("before moving:", t)
        print("after moving:", s)
        print("moved grad:", t.grad)


    def test_drawbox(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the X, Y, and Z axes
        P3.box(arr([1, 2, 1], dtype = float),
                    size = 5, ax = ax)
        P3.coordinates((0,0,0), 4, ax = ax)

        ax.grid(False)
        ax.axis("off")

        
    def test_done(self):
        # plt.show()
        pass