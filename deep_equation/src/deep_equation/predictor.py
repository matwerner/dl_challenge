"""
Predictor interfaces for the Deep Learning challenge.
"""

from typing import List
import numpy as np
import os
import torch

from .helper import ImagePreprocessing, DeepEquationOutputHelper
from .model import DeepEquationNet

class BaseNet:
    """
    Base class that must be used as base interface to implement 
    the predictor using the model trained by the student.
    """

    def load_model(self, model_path):
        """
        Implement a method to load models given a model path.
        """
        pass

    def predict(
        self, 
        images_a: List, 
        images_b: List, 
        operators: List[str], 
        device: str = 'cpu'
    ) -> List[float]:
        """
        Make a batch prediction considering a mathematical operator 
        using digits from image_a and image_b.
        Instances from iamges_a, images_b, and operators are aligned:
            - images_a[0], images_b[0], operators[0] -> regards the 0-th input instance
        Args: 
            * images_a (List[PIL.Image]): List of RGB PIL Image of any size
            * images_b (List[PIL.Image]): List of RGB PIL Image of any size
            * operators (List[str]): List of mathematical operators from ['+', '-', '*', '/']
                - invalid options must return `None`
            * device: 'cpu' or 'cuda'
        Return: 
            * predicted_number (List[float]): the list of numbers representing the result of the equation from the inputs: 
                [{digit from image_a} {operator} {digit from image_b}]
        """
    # do your magic

    pass 


class RandomModel(BaseNet):
    """This is a dummy random classifier, it is not using the inputs
        it is just an example of the expected inputs and outputs
    """

    def load_model(self, model_path):
        """
        Method responsible for loading the model.
        If you need to download the model, 
        you can download and load it inside this method.
        """
        np.random.seed(42)

    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ) -> List[float]:

        predictions = []
        for image_a, image_b, operator in zip(images_a, images_b, operators):
            random_prediction = np.random.uniform(-10, 100, size=1)[0]
            predictions.append(random_prediction)
        
        return predictions


class StudentModel(BaseNet):
    """
    TODO: THIS is the class you have to implement:
        load_model: method that loads your best model.
        predict: method that makes batch predictions.
    """

    def __init__(self) -> None:
        super().__init__()
        source_dir = os.path.abspath(os.path.dirname(__file__))
        resource_dir = source_dir + '/../../resources'
        self.load_model(resource_dir + '/deep_equation_net.pt')
        self.operator_map = {'+': 0, '-': 1, '*': 2, '/': 3}

    # TODO
    def load_model(self, model_path: str):
        """
        Load the student's trained model.
        TODO: update the default `model_path` 
              to be the correct path for your best model!
        """
        self.model = DeepEquationNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()        
    
    # TODO:
    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ):
        """Implement this method to perform predictions 
        given a list of images_a, images_b and operators.
        """
        temp_model = self.model.to(device)
        predictions = []
        for image_a, image_b, operator in zip(images_a, images_b, operators):
            op = operator

            # Preprocessing
            image_a = ImagePreprocessing.preprocess_rgb_image(image_a).unsqueeze(0).to(device)
            image_b = ImagePreprocessing.preprocess_rgb_image(image_b).unsqueeze(0).to(device)
            operator = self.operator_map[operator]
            operator = torch.tensor([operator]).unsqueeze(0).to(device)

            # Run model
            output_a, output_b, output_eq = temp_model(image_a, image_b, operator)

            # Posprocessing
            output_a = output_a.detach().numpy().argmax()
            output_b = output_b.detach().numpy().argmax()
            output_eq = output_eq.detach().numpy().argmax()
            output_eq = DeepEquationOutputHelper.class_to_value(output_eq)

            predictions.append(output_eq)
            # print(output_a, output_b, op, output_eq)

        return predictions
