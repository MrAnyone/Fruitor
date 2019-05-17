import cv2
from PIL import Image
import torch
import numpy as np
from torchvision import transforms, datasets, models
from torch.optim import lr_scheduler
from torch import nn, optim
import time
import copy

means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # img = cv2.imread(image, cv2.IMREAD_COLOR)
    img = Image.open(image)

    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, std)])
    img = transformations(img)

    np_image = np.array(img)

    return np_image


data = torch.load('./fruitor_model_linear.pt')

_class = data['class_to_index']

# image = process_image('./datasets/validation_set/Cocos/13_100.jpg')
image = process_image('./datasets/test_set/raspberry-2584375.jpg')
image = torch.from_numpy(image)
image = image.to(device)
#
densenet = models.densenet201(pretrained=True)
densenet.classifier = nn.Linear(data['classifier_input_size'], len(data['class_to_index']))
densenet.to(device)
densenet.load_state_dict(data['state_dict'])

densenet.eval()

image.unsqueeze_(0)

output = densenet.forward(image)

ps = torch.exp(output)

probs, classes = ps.topk(5)

classes = classes.tolist()[0]
# print(_class[classes[0]])
for key, value in _class.items():
    if value == classes[0]:
        print(key)
        break

