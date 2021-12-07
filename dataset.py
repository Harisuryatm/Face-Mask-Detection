import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


# Custom Dataset for the pytoch DataLoader to create batches
class FaceMaskDataset(Dataset):
  """
  Custom Dataset for the pytoch DataLoader to create batches
  """

  def __init__(self, data_folder):
    self.data_folder= data_folder

    # opening the json files of training
    with open(os.path.join(data_folder, "TRAIN_images.json"), "r") as j:
      self.images= json.load(j)
    with open(os.path.join(data_folder, "TRAIN_objects.json"), "r") as j:
      self.objects= json.load(j)

    assert len(self.images) == len(self.objects)

  
  def __getitem__(self,i):
    # reading image
    image= Image.open(self.images[i], mode= 'r')
    image= image.convert("RGB")

    # reading objects (bbox, labels) in this image
    objects= self.objects[i]
    boxes= torch.FloatTensor(objects["boxes"])
    labels= torch.LongTensor(objects["labels"])

    # applying custom transformation
    image, boxes, labels= transform(image, boxes, labels)

    return image, boxes, labels
  
  def __len__(self):
    return len(self.images)
  
  def collate_fn(self, batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    Note: this need not be defined in this Class, can be standalone.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels
    """

    images = list()
    boxes = list()
    labels = list()

    for b in batch:
      images.append(b[0])
      boxes.append(b[1])
      labels.append(b[2])

    images = torch.stack(images, dim=0)

    return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each

    
