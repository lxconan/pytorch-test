import unittest
import torch
from pathlib import Path
from linear_regression_model_to_save import LinearRegressionModelToSave
from linear_regression_model_trainer import LinearRegressionModelTrainer


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.bias = 0.5
        self.weight = 3.0
        self.model = LinearRegressionModelToSave()

    def test_saving_model_state_for_trained_model(self):
        train_set_x, train_set_y = self.create_training_set()
        self.train_model(train_set_x, train_set_y)
        print(f'The trained model state: {self.model.state_dict()}')

        # 1. We can save the model we trained in a Python pickle file. This is a very simple way to save the model.
        #    To Do this operation, we can use the `torch.save()` method.
        # 2. We can load the model we trained from a pickle file using `torch.load()` method.
        # 3. We can use `torch.nn.Module.load_state_dict()` allows us to load the model's state dictionary.
        #
        # The recommended way to save and load the model is to save and load the state dictionary.
        #
        # For more information about saving and loading model in PyTorch, please refer to this link:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        model_path = Path('saved_models')
        model_path.mkdir(parents=True, exist_ok=True)
        model_name = 'linear_regression_model.pt'
        model_save_path = model_path / model_name
        print(f'We will save the model to: "{model_save_path}"')
        torch.save(self.model.state_dict(), model_save_path)

        # Let's load the model back to model
        another_model = LinearRegressionModelToSave()
        another_model.load_state_dict(torch.load(model_save_path))
        print(f'Model state loaded: {another_model.state_dict()}')

    def create_training_set(self):
        train_set_x = torch.arange(0., 1., 0.02).unsqueeze(dim=1)
        train_set_y = self.weight * train_set_x + self.bias
        return train_set_x, train_set_y

    def train_model(self, train_set_x, train_set_y):
        trainer = LinearRegressionModelTrainer(self.model)
        trainer.train(10000, 0.00000001, train_set_x, train_set_y)


if __name__ == '__main__':
    unittest.main()
