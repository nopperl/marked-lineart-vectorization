import unittest

import torch

from marked_lineart_vec.models import IterativeModel


class IterativeModelTestCase(unittest.TestCase):
    model = None

    def setUp(self) -> None:
        self.model = IterativeModel()

    def test_vector_loss_sanity(self):
        bs, num_curves, num_points = 4, 3, 3
        bezier_points = torch.rand((bs, num_points, 2))
        input_points = torch.rand((bs, num_curves, num_points, 2))
        same_input_points = torch.repeat_interleave(bezier_points.unsqueeze(1), num_curves, dim=1)
        # L(X, X) = 0
        self.assertEqual(self.model.loss_function(None, bezier_points=bezier_points, input_points=same_input_points)["loss"], 0)
        # L(X,Y) = L(Y,X)
        self.assertEqual(self.model.loss_function(None, bezier_points=bezier_points, input_points=input_points[:, 0].unsqueeze(1))["loss"],
                         self.model.loss_function(None, bezier_points=input_points[:, 0], input_points=same_input_points)["loss"])
        # Correct values
        fixed_bezier_points = torch.tensor([[[2., 3.], [1., 4.]], [[5., 2.], [6., 7.]]])
        fixed_input_points = torch.tensor([[[[4., 6.], [4., 5.]], [[3., 7.], [2., 1.]]], [[[2., 3.], [5., 2.]], [[4., 1.], [6., 3.]]]])
        self.assertEqual(self.model.loss_function(None, bezier_points=fixed_bezier_points, input_points=fixed_input_points)["loss"], 5.125)
        zero_bezier_points = torch.zeros_like(fixed_bezier_points)
        self.assertEqual(self.model.loss_function(None, bezier_points=zero_bezier_points, input_points=fixed_input_points)["loss"], 13.125)
        zero_input_points = torch.zeros_like(fixed_input_points)
        self.assertEqual(self.model.loss_function(None, bezier_points=fixed_bezier_points, input_points=zero_input_points)["loss"], 18.0)


if __name__ == '__main__':
    unittest.main()
