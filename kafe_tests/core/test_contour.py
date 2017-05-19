import unittest
from kafe.core.contour import ContourFactory

class TestContour(unittest.TestCase):
    
    def setUp(self):
        self._contour_1_a = ContourFactory.create_xy_contour([
            [-1.0,  0.0,  1.0, 1.0, 1.0, 0.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, 0.0, 1.0, 1.0,  1.0,  0.0]], 1.0)
        self._contour_1_b = ContourFactory.create_xy_contour([
            [-0.5,  0.5,  1.0, 1.0, 0.5, -0.5, -1.0, -1.0],
            [-1.0, -1.0, -0.5, 0.5, 1.0,  1.0,  0.5, -0.5]],1.0)
        self._contour_1_c = ContourFactory.create_xy_contour([
            [-10.0,   0.0,  10.0, 10.0, 10.0,  0.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0,  0.0, 10.0, 10.0,  10.0,   0.0]], 1.0)
        
    def test_identity(self):
        self.assertTrue(self._contour_1_a.is_similar_to(self._contour_1_a))
        self.assertTrue(self._contour_1_b.is_similar_to(self._contour_1_b))
        
    def test_1_a_b_equal(self):
        self.assertTrue(self._contour_1_a.is_similar_to(self._contour_1_b))
        
    def test_1_a_c_not_equal(self):
        self.assertFalse(self._contour_1_a.is_similar_to(self._contour_1_c))
        self.assertFalse(self._contour_1_c.is_similar_to(self._contour_1_a))
        