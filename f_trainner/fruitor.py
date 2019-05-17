import torch
from torchvision import transforms, datasets, models
from torch.optim import lr_scheduler
from torch import nn, optim
import time
import copy

checkpoint_path = './fruitor_checkpoint.pth'

# device = 'cpu'
# if torch.cuda.is_available:
#     print('cuda ok')
#     device = 'cuda'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dir = './datasets/training_set'
validation_dir = './datasets/validation_set'

means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'training': transforms.Compose([transforms.Resize(256),
                                    transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, std)]),
    'validation': transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(means, std)]),
}

image_datasets = {
    'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
    'validation': datasets.ImageFolder(validation_dir, transform=data_transforms['validation'])
}

dataloaders = {
    'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=128, shuffle= True),
    'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32, shuffle= True)
}

class_to_index = image_datasets['training'].class_to_idx
dataset_sizes = {
    'training': len(image_datasets['training']),
    'validation': len(image_datasets['validation'])
}
class_numbers = len(class_to_index)

densenet = models.densenet201(pretrained=True)

for param in densenet.parameters():
    param.requires_grad = False

classifier_input_size = densenet.classifier.in_features

# densenet.classifier = nn.Sequential(
#                       # nn.Linear(classifier_input_size, class_numbers),
#                       nn.Linear(classifier_input_size, 256),
#                       nn.ReLU(),
#                       nn.Dropout(0.4),
#                       nn.Linear(256, class_numbers),
#                       nn.LogSoftmax(dim=1))

densenet.classifier = nn.Linear(classifier_input_size, class_numbers)

densenet.cuda()


def train_model(data_set, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            counter = 0

            # Iterate over data.
            for inputs, labels in data_set[phase]:
                print('\r({} - {})'.format(counter, '52'), end='')

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    checkpoint = {
                  'classifier_input_size': classifier_input_size,
                  'class_to_index': class_to_index,
                  'state_dict': densenet.state_dict()}
    torch.save(checkpoint, './fruitor_model_linear.pt')
    print('Checkpoint saved')
    return model


optimizer_ft = optim.SGD(densenet.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model = train_model(dataloaders, densenet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)