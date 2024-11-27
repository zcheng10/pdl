import unittest
import sys

sys.path.append("../")
from src.pnn import *
from src.illustrate import *

class TestBasic(unittest.TestCase):

    def test_Object(self):
        """Test Object"""
        a = Object("ball", size = 0.1, 
                   r = np.array([10, 0, 2]) )
        p = Projector(hs = 0.2, h0 = arr([1, 0, 0]), h1 = arr([0, 1, 0]))
        br = p.toScreen(a)
        print("a =", a)
        print("br =", br)

        # self.assertEqual(square(3), 9)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        P3.plot_box(a.r, a.size)
        P3.plot_box(arr([0,0,0]), size=0.05)
        P3.plot_box(-p.h0 * p.hs, size = 2)
        ax.axis("equal")
        plt.show()

    def test_drawbox(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the X, Y, and Z axes
        P3.plot_box(arr([1, 2, 1], dtype = float),
                    size = 5)

        ax.grid(False)
        ax.axis("off")
        