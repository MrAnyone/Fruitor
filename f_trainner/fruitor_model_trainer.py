import torch
from torchvision import transforms, datasets, models
from torch.optim import lr_scheduler
from torch import nn, optim
import numpy as np
import time
import copy
from PIL import Image


class FTrainer:
    TRAINING_SET_PATH = './datasets/training_set'
    VALIDATION_SET_PATH = './datasets/validation_set'
    MEANS = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = {
            'training': transforms.Compose([transforms.Resize(256),
                                            transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.MEANS, self.STD)]),
            'validation': transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.MEANS, self.STD)]),
        }
        self.image_datasets = None
        self.dataloaders = None
        self.class_to_index = None
        self.dataset_sizes = None
        self.class_numbers = None
        self.densenet = None

    def load_set(self, training_set=None, validation_set=None):
        self.image_datasets = {
            'training': datasets.ImageFolder(training_set if training_set else self.TRAINING_SET_PATH, transform=self.data_transforms['training']),
            'validation': datasets.ImageFolder(validation_set if validation_set else self.VALIDATION_SET_PATH, transform=self.data_transforms['validation'])
        }
        self.dataset_sizes = {
            'training': len(self.image_datasets['training']),
            'validation': len(self.image_datasets['validation'])
        }
        self.dataloaders = {
            'training': torch.utils.data.DataLoader(self.image_datasets['training'], batch_size=300, shuffle=True),
            'validation': torch.utils.data.DataLoader(self.image_datasets['validation'], batch_size=30, shuffle=True)
        }
        self.class_to_index = self.image_datasets['training'].class_to_idx
        self.class_numbers = len(self.class_to_index)

    def train(self, parameters):
        since = time.time()

        self.densenet = models.densenet201(pretrained=True)

        for param in self.densenet.parameters():
            param.requires_grad = False

        if not self.dataloaders:
            self.load_set()
        classifier_input_size = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(classifier_input_size, self.class_numbers)
        self.densenet = self.densenet.to(self.device)
        optimizer = optim.SGD(self.densenet.parameters(), lr=0.001, momentum=parameters['momentum'])
        criterion = nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        best_model_wts = copy.deepcopy(self.densenet.state_dict())
        best_acc = 0.0

        sub_since = time.time()

        for epoch in range(parameters['num_epochs']):
            print('Epoch {}/{}'.format(epoch, parameters['num_epochs'] - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['training', 'validation']:
                if phase == 'training':
                    scheduler.step()
                    self.densenet.train()  # Set model to training mode
                else:
                    self.densenet.eval()  # Set model to evaluate mode

                running_loss = 0.0
                counter = 0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    # print('ok')
                    print('\r({} ~ {} => )'.format(counter,
                                                   39 if phase == 'training' else 66,
                                                   time.time() - sub_since
                                                   ), end='')
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'training'):
                        outputs = self.densenet(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'training':
                            loss.backward()
                            optimizer.step()
                    counter += 1

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'validation' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.densenet.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.densenet.load_state_dict(best_model_wts)
        checkpoint = {
            'classifier_input_size': classifier_input_size,
            'class_to_index': self.class_to_index,
            'state_dict': self.densenet.state_dict()}
        torch.save(checkpoint, './fruitor_model_linear.pt')
        print('Checkpoint saved')
        print('Model Ready to predict')

    def predict(self, img_brut):
        self.densenet.eval()
        pil_img = Image.fromarray(img_brut)
        img = self.data_transforms['validation'](pil_img)
        image = torch.from_numpy(np.array(img))
        image = image.to(self.device)
        image.unsqueeze_(0)
        output = self.densenet.forward(image)
        ps = torch.exp(output)
        probs, classes = ps.topk(5)

        result = []
        classes = classes.tolist()[0]
        for index in classes:
            for key, value in self.class_to_index.items():
                if value == index:
                    result.append(key)
        return result

    def load(self, path='./f_trainner/fruitor_densenet201_model_linear_class.pt'):
        data = torch.load(path)
        self.class_to_index = data['class_to_index']
        self.densenet = models.densenet201(pretrained=True)
        self.densenet.classifier = nn.Linear(data['classifier_input_size'], len(data['class_to_index']))
        self.densenet = self.densenet.to(self.device)
        self.densenet.load_state_dict(data['state_dict'])
        print('Model loaded => {}'.format(path))
        print('Model Ready to predict')
