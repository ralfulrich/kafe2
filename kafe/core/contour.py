import numpy as np
class ContourException(Exception):
    pass

class ContourFactory():
    
    @staticmethod
    def create_xy_contour(xy_points, sigma):
        xy_points = np.asarray(xy_points)
        _shape = xy_points.shape
        if len(_shape) != 2 or (_shape[0] != 2 and _shape[1] != 2):
            raise ContourException("Explicit contours can only be created from iterables with shape (2,n) or (n,2).")
        if _shape[0] != 2:
            xy_points = xy_points.T
        return Contour(xy_points=xy_points, sigma=sigma)
    
    @staticmethod
    def create_grid_contour(grid_x, grid_y, grid_z, sigma):
        grid_x = np.asarray(grid_x)
        grid_y = np.asarray(grid_y)
        grid_z = np.asarray(grid_z)
        _shape_x = grid_x.shape
        _shape_y = grid_y.shape
        _shape_z = grid_z.shape
        if len(_shape_x) != 1:
            raise ContourException("grid_x needs to be one-dimensional.")
        if len(_shape_y) != 1:
            raise ContourException("grid_y needs to be one-dimensional.")
        if len(_shape_z) != 2:
            raise ContourException("grid_z needs to be two-dimensional.")
        if _shape_x[0] != _shape_z[0]:
            raise ContourException("grid_z needs to be as wide as grid_x is long.")
        if _shape_y[0] != _shape_z[1]:
            raise ContourException("grid_z needs to be as high as grid_y is long.")
        return Contour(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, sigma=sigma)
    

class Contour(object):
    
    def __init__(self, xy_points=None, grid_x=None, grid_y=None, grid_z=None, sigma=None):
        if sigma is None:
            raise ContourException("sigma must not be None.")
        self._xy_points = xy_points
        self._grid_x = grid_x
        self._grid_y = grid_y
        self._grid_z = grid_z
        self._sigma = sigma

    @property
    def xy_points(self):
        return self._xy_points
    
    @property
    def grid_x(self):
        return self._grid_x
    
    @property
    def grid_y(self):
        return self._grid_y
    
    @property
    def grid_z(self):
        return self._grid_z
    
    @property
    def sigma(self):
        return self._sigma
    
    def is_similar_to(self, another_contour):
        if another_contour is None:
            raise ContourException("another_contour must not be None.")
        if self.xy_points is None:
            if another_contour.xy_points is None:
                return Contour._compare_grid_to_grid(self, another_contour)
            else:
                return Contour._compare_grid_to_xy(self, another_contour)
        else:
            if another_contour.xy_points is None:
                return Contour._compare_grid_to_xy(another_contour, self)
            else:
                return Contour._compare_xy_to_xy(another_contour, self)

    @staticmethod
    def _compare_xy_to_xy(xy_contour_1, xy_contour_2):
        _length_1 = xy_contour_1.xy_points.shape[1]
        _length_2 = xy_contour_2.xy_points.shape[1]
        print _length_1, _length_2
        if _length_1 > _length_2:
            _coarse_contour = xy_contour_2.xy_points
            _coarse_length = _length_2
            _fine_contour = xy_contour_1.xy_points
            _fine_length = _length_1
        else:
            _coarse_contour = xy_contour_1.xy_points
            _coarse_length = _length_1
            _fine_contour = xy_contour_2.xy_points
            _fine_length = _length_2

        _coarse_points = _coarse_contour.T
        _fine_points = _fine_contour.T

        _coarse_index = 0
        _squared_distances = (_fine_contour[0] - _coarse_points[0,0]) ** 2 + (_fine_contour[1] - _coarse_points[0,1]) ** 2
        _min_index = np.argmin(_squared_distances)
        _goal_point_fine = _fine_points[_min_index - 1]
        _goal_point_coarse = _coarse_points[-1]
        
        _current_point = _fine_points[_min_index - 1]
        _fine_index = _min_index
        _coarse_index = 0
        
        _reached_goal_fine = False
        _reached_goal_coarse = False
        
        count = 0
        while True:
            count += 1
            if np.sum((_current_point - _fine_points[_fine_index]) ** 2) < np.sum((_current_point - _coarse_points[_coarse_index]) ** 2):
                if _reached_goal_fine:
                    return False
                _current_point = _fine_points[_fine_index]
                _fine_index = (_fine_index + 1) % _fine_length
                print count, "fine", _current_point
                if _current_point[0] == _goal_point_fine[0] and _current_point[1] == _goal_point_fine[1]:
                    _reached_goal_fine = True

            else:
                if _reached_goal_coarse:
                    return False
                _current_point = _coarse_points[_coarse_index]
                _coarse_index = (_coarse_index + 1) % _coarse_length
                print count, "coarse", _current_point
                if _current_point[0] == _goal_point_coarse[0] and _current_point[1] == _goal_point_coarse[1]:
                    _reached_goal_coarse = True

            if _reached_goal_fine and _reached_goal_coarse:
                return True
            
    @staticmethod
    def _compare_grid_to_xy(grid_contour, xy_contour):
        return False
    
    @staticmethod
    def _compare_grid_to_grid(grid_contour_1, grid_contour_2):
        return False
    
    