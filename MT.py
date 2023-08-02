import random

import torch
import torchvision
import torchvision.transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision.models.optical_flow import Raft_Large_Weights
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class MR():
    def __init__(self, name):
        self.name = name

    def SetM(self, m):
        self.m = m

    def GetM(self):
        return self.m

    def ToPrint(self):
        return self.name

class Rotate(MR):
    def __init__(self, angle = 50):
        super(Rotate, self).__init__("Rotate {} deg".format(angle))
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)

class Invert(MR):
    def __init__(self):
        super(Invert,self).__init__( "Inverting")

    def __call__(self, x):
        return TF.invert(x)

class Scale(MR):
    def __init__(self, scale=0.9):
        super(Scale, self).__init__("Scaling {}%".format(scale*100))
        self.scale=scale

    def __call__(self, x):
        return TF.affine(x, scale=self.scale , translate=[0,0], shear=0,angle=0)

class HorizontalTranslation(MR):
    def __init__(self, translation=2):
        super(HorizontalTranslation, self).__init__("H Trans by {}".format(translation))
        self.translation=translation

    def __call__(self, x):
        return TF.affine(x, scale=1.0 , translate=[self.translation,0], shear=0,angle=0)

class VerticalTranslation(MR):
    def __init__(self, translation=2):
        super(VerticalTranslation, self).__init__("V Trans by {}".format(translation))
        self.translation=translation

    def __call__(self, x):
        return TF.affine(x, scale=1.0 , translate=[0, self.translation], shear=0,angle=0)

class Shear(MR):
    def __init__(self, shear=2):
        super(Shear, self).__init__("Shear by {}".format(shear))
        self.shear=shear

    def __call__(self, x):
        return TF.affine(x, scale=1.0 , translate=[0, 0], shear=self.shear,angle=0)

class Ploter():
    def __init__(self,samples, dataset, tf_list):
        self.samples = samples
        self.dataset = dataset
        self.tf_list = tf_list

    def getASampleImagesAndItsTrans(self, idx):
        images = [self.dataset[idx][0]]   # Dataset[i] will return a tuple of (img, label)
        tf_dataset_list = [transformed_dataset(self.dataset, tf) for tf in self.tf_list]
        for tf_ds in tf_dataset_list:
            images.append(tf_ds[idx][0])
        return images

    def assertWhetherIthLabelIsSampled(self, idx):
        _, label = self.dataset[idx]
        if not label in self.sampled_labels:
            self.sampled_labels.append(label)
            return True
        else:
            return False

    def getSampleImages(self):
        sampled_images = []
        self.sampled_labels = []
        i = 0
        while(True):
            if len(self.sampled_labels)>=10:
                break
            if self.assertWhetherIthLabelIsSampled(i):
                sample_i_and_its_trans = self.getASampleImagesAndItsTrans(i)
                sampled_images.append(sample_i_and_its_trans)
            i+=1
        return sampled_images

    def getPreix(self):
        tf_prefix = [tf.ToPrint() for tf in tf_list]
        label_prefix = ['Original']+tf_prefix
        return label_prefix

    def Plot(self):
        sampled_images = self.getSampleImages()
        tags_of_trans = self.getPreix()
        rows = len(sampled_images)
        cols = len(sampled_images[0])
        fig,axes =  plt.subplots(rows, cols, figsize=(11,15))
        for r in range(rows):
            for c in range(cols):
                ax = axes[r, c]
                ax.axis('off')
                img = sampled_images[r][c]
                if r == 0:                         #This is to set the first row's title
                    ax.set_title(tags_of_trans[c])
                ax.imshow(img.squeeze(), cmap = 'gray')
        plt.show()

class transformed_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item, label = self.dataset[idx]
        item = self.transform(item)
        return item,label



def mt(model, device, test_loader, transforms):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            source_input, target = data.to(device), target.to(device)
            source_output = model(source_input)
            follow_up_input = transforms(source_input)
            follow_up_output = model(follow_up_input)
            source_pred = source_output.argmax(dim = 1, keepdim=True)
            follow_up_pred = follow_up_output.argmax(dim=1, keepdim=True)
            correct += source_pred.eq(follow_up_pred.view_as(source_pred)).sum().item()
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
    #test_loss /= len(test_loader.dataset)
    mts = 100.0*correct/len(test_loader.dataset)
    print('\nTest set: Metamorphic Testing Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),  mts))
    return mts

if __name__ == "__main__":
    mnist_train = datasets.MNIST('data', train=True, download=True,
                              transform=torchvision.transforms.ToTensor())
    tf_list = [Invert(), Rotate(10), Scale(0.4), HorizontalTranslation(5), VerticalTranslation(5), Shear(30)]
    tf_list = [Rotate(10),Rotate(1),Rotate(30)]
    samples = 10
    ImagePlot = Ploter(samples, mnist_train,tf_list)
    ImagePlot.Plot()