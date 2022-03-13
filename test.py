# Purpose: Test the saved model and show AUC ROC curve and percentage accuracy
# @author: Dominic Sobocinski

import torch
from torchvision import models
from skimage import io
from torchvision import transforms as T
import matplotlib.pyplot as plt
from sklearn import metrics
import shutil

# check to see what files were used for validation
# this is what we will use for testing the accuracy of the model since
# this data was not used to train the model
f = open("val_files.txt")
img_dir = []
#load in image labels
labels = []
for file in f:
    img_dir.append(file)
    folders = file.split('/')
    if folders[1] == 'background':
        labels.append(0)
    else:
        labels.append(1)
f.close()

# validate on gpu if we can
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dict = torch.load("Oyster_classifier.pth")  # load the model
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(dict['model'])
model.to(device)
model.eval()

results = []

# prepare the images to send them through the model
prep = T.Compose([T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# go through each image
for dir in img_dir:
    image = io.imread(dir.rstrip('\n'))
    image = prep(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    out = model(image)
    results.append(torch.argmax(out).item())

right = 0

for i in range(len(labels)):
    if results[i] == labels[i]:
        right += 1
    else:
        print(img_dir[i], results[i])
        shutil.copy(img_dir[i].strip('\n'), "results/"+str(i) + "_" + str(results[i]) + ".jpg")

print("Accuracy:", right * 100 / len(labels))

fpr, tpr, thresh = metrics.roc_curve(labels, results)
roc_auc = metrics.auc(fpr, tpr)

# roc curve
plt.title('ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
