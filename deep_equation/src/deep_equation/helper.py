import cv2
import math
import numpy as np
import torch
import random
from PIL import Image
from scipy import ndimage

# Reference: 
# 1. https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
# 2. https://github.com/opensourcesblog/tensorflow-mnist/blob/master/mnist.py
class ImagePreprocessing:

    @staticmethod
    def getBestShift(img):
        cy,cx = ndimage.measurements.center_of_mass(img)

        rows,cols = img.shape
        shiftx = np.round(cols/2.0-cx).astype(int)
        shifty = np.round(rows/2.0-cy).astype(int)

        return shiftx,shifty

    @staticmethod
    def shift(img,sx,sy):
        rows,cols = img.shape
        M = np.float32([[1,0,sx],[0,1,sy]])
        shifted = cv2.warpAffine(img,M,(cols,rows))
        return shifted

    # Reference
    # https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    @staticmethod
    def noise(img,prob):    
        '''
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
        output = np.zeros(img.shape)
        thres = 1 - prob 
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        return output

    @staticmethod
    def preprocess_mnist_image(pixels, add_noise=True):
        pixels = 255 * pixels.cpu().detach().numpy()
        image = Image.fromarray(pixels.squeeze(0))
        return ImagePreprocessing.preprocess_rgb_image(image, add_noise=add_noise)

    @staticmethod
    def preprocess_rgb_image(image, background_threshold=128, add_noise=False):
        # Greyscale
        image = image.convert('L')
        gray = np.array(image)

        # Rescale and convert background to black if needed
        value = np.median(np.array(image))        
        if value >= background_threshold:
            gray = cv2.resize(255 - gray, (28, 28))
        else:
            gray = cv2.resize(gray, (28, 28))
        # better black and white version
        value = np.median(np.array(gray))
        (_, gray) = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)

        while np.sum(gray[0]) == 0:
            gray = gray[1:]

        while np.sum(gray[:, 0]) == 0:
            gray = np.delete(gray, 0, 1)

        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]

        while np.sum(gray[:, -1]) == 0:
            gray = np.delete(gray, -1, 1)

        if add_noise:
            gray = ImagePreprocessing.noise(gray, 0.05)

        rows, cols = gray.shape

        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            # first cols than rows
            gray = cv2.resize(gray, (cols, rows))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            # first cols than rows
            gray = cv2.resize(gray, (cols, rows))

        colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
        rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
        gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

        shiftx, shifty = ImagePreprocessing.getBestShift(gray)
        shifted = ImagePreprocessing.shift(gray, shiftx, shifty)
        gray = shifted

        # -1 to 1
        gray = gray / 255.0
        gray = 2.0 * gray - 1.0

        # To tensor
        return torch.tensor(gray, dtype=torch.float32).unsqueeze(0)

class DeepEquationOutputHelper:
    values = ['nan', '-9.0', '-8.0', '-7.0', '-6.0', '-5.0', '-4.0', '-3.0', '-2.0', '-1.0', '0.0', '0.11', '0.12', '0.14', '0.17', '0.2', '0.22', '0.25', '0.29', '0.33', '0.38', '0.4', '0.43', '0.44', '0.5', '0.56', '0.57', '0.6', '0.62', '0.67', '0.71', '0.75', '0.78', '0.8', '0.83', '0.86', '0.88', '0.89', '1.0', '1.12', '1.14', '1.17', '1.2', '1.25', '1.29', '1.33', '1.4', '1.5', '1.6', '1.67', '1.75', '1.8', '2.0', '2.25', '2.33', '2.5', '2.67', '3.0', '3.5', '4.0', '4.5', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '11.0', '12.0', '13.0', '14.0', '15.0', '16.0', '17.0', '18.0', '20.0', '21.0', '24.0', '25.0', '27.0', '28.0', '30.0', '32.0', '35.0', '36.0', '40.0', '42.0', '45.0', '48.0', '49.0', '54.0', '56.0', '63.0', '64.0', '72.0', '81.0']
    value_to_class_map = {value:i for i, value in enumerate(values)}
    class_to_value_map = {i:value for i, value in enumerate(values)}
       
    @staticmethod
    def value_to_class(value):
        key = str(round(float(value), 2))
        return DeepEquationOutputHelper.value_to_class_map[key]

    @staticmethod
    def class_to_value(key):    
        value_str = DeepEquationOutputHelper.class_to_value_map[key]
        return float(value_str) if value_str != 'nan' else np.nan

class DeepEquationDataset(torch.utils.data.Dataset):

    def __init__(self, loader, size) -> None:
        super().__init__()
        #self.loader = loader
        self.size = size
        self.indices_a = np.random.randint(0, len(loader), size)
        self.indices_b = np.random.randint(0, len(loader), size)
        self.operators = np.random.randint(0, 4, size)
        """
        targets_a = [loader[index][1] for index in self.indices_a]
        targets_b = [loader[index][1] for index in self.indices_b]
        print(np.bincount(targets_a))
        print(np.bincount(targets_b))
        """

        self.images = [(ImagePreprocessing.preprocess_mnist_image(image), target) 
                        for image, target in loader]                

    def __getitem__(self, index):
        image_a, target_a = self.images[self.indices_a[index]]
        image_b, target_b = self.images[self.indices_b[index]]
        operator = self.operators[index]

        instance = (image_a, image_b, torch.tensor([operator]))
        if operator == 0:   # +
            target_eq = target_a + target_b
        elif operator == 1: # -
            target_eq = target_a - target_b
        elif operator == 2: # *
            target_eq = target_a * target_b
        else:               # /
            target_eq = target_a / target_b if target_b > 0 else np.nan
        target_eq = DeepEquationOutputHelper.value_to_class(target_eq)
        target = (target_a, target_b, torch.tensor(target_eq))
        return (instance, target)

    def __len__(self):
        return self.size