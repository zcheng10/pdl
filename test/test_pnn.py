import unittest
import sys

sys.path.append("../")
from src.pnn import *

class TestBasic(unittest.TestCase):

    def test_Object(self):
        """Test Object"""
        a = Object("ball", size = 0.1, 
                   r = np.array([10, 0, 2]) )
        p = Projector(hs = 0.2, h0 = arr([1, 0, 0]))
        # self.assertEqual(square(3), 9)