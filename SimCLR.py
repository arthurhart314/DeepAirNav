import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from lightly.data import LightlyDataset, SimCLRCollateFunction, collate

from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

import torchvision.transforms as T
import torchvision.transforms.v2 as  T2
from lightly.data import BaseCollateFunction

from lightly.transforms.simclr_transform import SimCLRTransform


def custom_collate_fn(input_size):
    T.ElasticTransform()
    #transform = [
    #         T.RandomHorizontalFlip(p=0.5),
    #         T.RandomRotation(degrees=(-180, 180)),
    #         T.RandomPerspective(p = 0.3),
    #         T2.RandomPhotometricDistort(p = 0.3),
    #         T.GaussianBlur((51,51)),
    #         T.RandomPosterize(4, p = 0.3),
    #         T.RandomCrop(300),
    #         T.Resize((input_size, input_size)),
    #         T.RandomEqualize(p = 0.3),
    #         T.RandomAutocontrast(p = 0.3),
    #         T.ColorJitter(),
    #         T2.RandomErasing(p =0.3),
    #         T.ToTensor(),
    #         T.Normalize(
    #            mean=collate.imagenet_normalize["mean"],
    #            std=collate.imagenet_normalize["std"],
    #        )]
    
    transform = [SimCLRTransform(), T.ElasticTransform()]

    transform = T.Compose(transform)

    collate_fn = BaseCollateFunction(transform) 

    return collate_fn


def test_transforms_fn(input_size):
    test_transforms = torchvision.transforms.Compose(
        [
            #torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=collate.imagenet_normalize["mean"],
                std=collate.imagenet_normalize["std"],
            ),
        ]
    )
    return test_transforms

def dataloader_train_simclr_fn(batch_size, num_workers, dataset_train_simclr, collate_fn):
    dataloader_train_simclr = torch.utils.data.DataLoader(
        dataset_train_simclr,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
    )
    return dataloader_train_simclr

def dataloader_test_fn(batch_size, num_workers, dataset_test):
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return dataloader_test


def train_dataloader_fn(data_folder, input_size, batch_size, num_workers):
    
    #collate_fn = custom_collate_fn(input_size)
    #collate_fn = SimCLRCollateFunction(input_size=input_size, random_gray_scale = 0.0, vf_prob=0.0, rr_prob=0.5, rr_degrees=(-180, 180))
    collate_fn = SimCLRCollateFunction(input_size=input_size, rr_degrees=(-180, 180))
    
    dataset_train_simclr = LightlyDataset(input_dir=data_folder)
    
    dataloader_train_simclr = dataloader_train_simclr_fn(batch_size, num_workers, dataset_train_simclr, collate_fn)

    return dataloader_train_simclr


def test_dataloader_fn(data_folder, input_size, batch_size, num_workers):

    test_transforms = test_transforms_fn(input_size)

    dataset_test = LightlyDataset(input_dir=data_folder, transform=test_transforms)

    dataloader_test = dataloader_test_fn(batch_size, num_workers, dataset_test)

    return dataloader_test

#def generate_dataloaders(config):
#
#    collate_fn = SimCLRCollateFunction(input_size=config['input size'], random_gray_scale = 0.0, vf_prob=0.0, rr_prob=0.5, rr_degrees=(-5, 5))
#
#    test_transforms = test_transforms_fn(config)
#
#    dataset_train_simclr = LightlyDataset(input_dir=config['train data folder'])
#
#    dataset_test = LightlyDataset(input_dir=config['test data folder'], transform=test_transforms)
#
#    dataloader_train_simclr = dataloader_train_simclr_fn(config, dataset_train_simclr, collate_fn)
#
#    dataloader_test = dataloader_test_fn(config, dataset_test)
#
#    return dataloader_train_simclr, dataloader_test


class SimCLRModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr          = self.config['learning rate'], 
            momentum    = self.config['momentum'], 
            weight_decay= self.config['weight decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.config['max epochs'])
        return [optim], [scheduler]
    
    
def load_saved_model(config):
    model = SimCLRModel(config)
    model.load_state_dict(torch.load(config['weights filepath']))
    return model


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)


def get_single_embedding(config, model, img):
    
    if type(img) != Image.Image:
        #print ('Converted to PIL.Image')
        img = Image.fromarray(img)
    
    test_transforms = test_transforms_fn(config['input size'])
    img_tensor = test_transforms(img)
    img_tensor = img_tensor[None, :]
    img_tensor = img_tensor.to(model.device)

    with torch.no_grad():
        emb = model.backbone(img_tensor).flatten(start_dim=1)
        emb = normalize(emb)

    return emb


def get_embeddings_from_img_arr(config, model, img_arr):

    #img_tensor = torch.Tensor(img_arr)
    
    #torch_dataset = torch.
    
    #test_transforms = test_transforms_fn(config)

    #dataset_test = LightlyDataset.from_torch_dataset(torch_dataset, transform=test_transforms)
    
    #dataloader_test = dataloader_test_fn(config, dataset_test)

    return 0


def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """
    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, label, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames


def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)


def plot_knn_examples(embeddings, filenames, path_to_data, n_neighbors=3, num_examples=6):###############
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    samples_idx = np.array(range(num_examples))#np.random.choice(len(indices), size=num_examples, replace=False)

    names = {}

    # loop through our randomly picked samples
    for idx in samples_idx:
        names[filenames[idx]] = []
        fig = plt.figure()
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # add the subplot
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            # get the correponding filename for the current index
            fname = os.path.join(path_to_data, filenames[neighbor_idx])
            
            names[filenames[idx]].append(filenames[neighbor_idx])
            
            # plot the image
            plt.imshow(get_image_as_np_array(fname))
            # set the title to the distance of the neighbor
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            #ax.set_title(filenames[idx])
            # let's disable the axis
            plt.axis("off")

    return names
        

