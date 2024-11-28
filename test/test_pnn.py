import unittest
import sys

sys.path.append("../")
from src.pnn import *
from src.illustrate import *

class TestBasic(unittest.TestCase):

    def test_Object(self):
        """Test Object"""
        obs = [Object("ball1", size = 0.1, 
                   r = np.array([10, 0, 2]) ),
            Object("ball2", size = 0.1, 
                   r = np.array([10, 0, 10])),
            Object("ball3", size = 0.1, 
                   r = arr([10, 8, 10]))]
        
        p = Projector(hs = 0.2, h0 = arr([1, 0, 0]), h1 = arr([0, 1, 0]))
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
        plt.show()

    def test_drawbox(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the X, Y, and Z axes
        P3.box(arr([1, 2, 1], dtype = float),
                    size = 5, ax = ax)
        P3.coordinates((0,0,0), 4, ax = ax)

        ax.grid(False)
        ax.axis("off")
        