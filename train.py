from Dataset import OysterMushroom
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
import os


writer = SummaryWriter()

# create the dataset
# form list of directories and targets
mush_dir = "preprocessed/oyster/"
non_mush_dir = "preprocessed/background/"
img_dir = []
img_labels = []

for file in os.listdir(mush_dir):
    img_labels.append(1)
    img_dir.append(mush_dir + file)

# add non oyster mushrooms and empty images to data
for file in os.listdir(non_mush_dir):
    img_labels.append(0)
    img_dir.append(non_mush_dir + file)

random_seed = 42  # random seed

train_size = 0.8  # split dataset into training set and validation set

train_dir, test_dir = train_test_split(
    img_dir,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

train_labels, test_labels = train_test_split(
    img_labels,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

# save test files to .txt
out = open("test_files.txt", "w")
for file in test_dir:
    out.write(file + '\n')

transformations = transforms.Compose([transforms.ToTensor(),
                                     transforms.GaussianBlur(kernel_size = (5, 9), sigma=(0.1, 5)),
                                     transforms.RandomRotation(degrees = (0, 12)),
                                     transforms.RandomHorizontalFlip(p = 0.3),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])

train_data = OysterMushroom(train_dir, train_labels, transform = transformations)  # train dataset
test_data = OysterMushroom(test_dir, test_labels, transform = transformations)  # test dataset

# create data loaders for easy training and testing
dataloader_train = DataLoader(dataset=train_data, batch_size = 16, shuffle = True)
dataloader_test = DataLoader(dataset=test_data, batch_size = 16, shuffle = True)


# initialize model and hyper parameters
# just going to use pre-trained MobileNet v2
# use cuda if we can
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = models.resnet50(pretrained = True)
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),  lr = 0.00001)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.995)

epochs = 1000
epoch = 0
# train the model
for epoch in range(epochs):
    avg_loss = 0
    print("\nepoch", epoch, ": train")
    model.train()
    for batch, loader in enumerate(dataloader_train):
        data = loader[0].to(device)
        target = loader[1].to(device)

        out = model(data)
        loss = loss_fn(out, target)
        loss.backward()
        avg_loss = avg_loss + loss.item()
        optimizer.step()

    writer.add_scalar('Loss/train', avg_loss / len(dataloader_train), epoch)
    print(epoch, "/", epochs, "loss:", avg_loss / len(dataloader_train))

    lr_scheduler.step()
    print(optimizer.param_groups[0]['lr'])

    # validate
    print("epoch", epoch, ": validation")
    model.eval()
    avg_loss = 0
    for batch, loader in enumerate(dataloader_test):

        data = loader[0].to(device)
        target = loader[1].to(device)
        out = model(data)
        loss = loss_fn(out, target)
        avg_loss = avg_loss + loss.item()

    #lr_scheduler.step(avg_loss / len(dataloader_test))

    writer.add_scalar('Loss/test', avg_loss / len(dataloader_test), epoch)
    print(epoch, "/", epochs, "loss:", avg_loss/ len(dataloader_test))

    if ( epoch % 5 == 0):
        torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict()},
                    "Oyster_classifier.pth")

writer.close()