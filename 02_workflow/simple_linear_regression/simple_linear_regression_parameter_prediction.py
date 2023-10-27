import unittest

from linear_regression_recognition import LinearRegressionRecognition


class TestPrepareAndLoadData(unittest.TestCase):
    # Data can be almost anything for machine learning. It can be images, text, audio, video, tabular data, etc.
    # Machine learning is a game of 2 parts.
    #
    # 1. Get data into numerical representation.
    # 2. Build a model that can learn from the numerical representation of the data.
    #
    # This is the first part of the game. We will learn how to get data into numerical representation. We start from
    # the linear regression problem. We will use a linear regression formular to make straight line from *known* data.

    def test_do_prediction_to_linear_regression_problem(self):
        linear_regression = LinearRegressionRecognition(actual_weight=2.0, actual_bias=1.0)
        linear_regression.train_and_predict()
