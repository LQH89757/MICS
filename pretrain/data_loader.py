

import os

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset



# class CustomDataset(Dataset):
#   def __init__(self, data_dir, class_name=None):
#     self.data_dir = data_dir
#     self.transform_WLI = transforms.Compose([
#       transforms.Resize((384, 384)),
#       transforms.ToTensor()
#     ])
#     self.transform_NBI = transforms.Compose([
#       transforms.Resize((384, 384)),
#       transforms.ToTensor()
#     ])
#     self.c_name = class_name
#     self.image_pairs = self._get_image_pairs()
#
#   def __len__(self):
#     return len(self.image_pairs)
#
#   def __getitem__(self, index):
#     wli_path, nbi_path = self.image_pairs[index]
#     wli_image = Image.open(wli_path).convert('RGB')
#     nbi_image = Image.open(nbi_path).convert('RGB')
#
#     if self.transform_WLI:
#       wli_image = self.transform_WLI(wli_image)
#     if self.transform_NBI:
#       nbi_image = self.transform_NBI(nbi_image)
#
#     return wli_image, nbi_image, index
#
#   def _get_image_pairs(self):
#     image_pairs = []
#     foder_name = [s.strip() for s in self.c_name.split(",")]
#     for category in foder_name:
#       wli_folder = os.path.join(self.data_dir, f"{category}_WLI")
#       nbi_folder = os.path.join(self.data_dir, f"{category}_NBI")
#
#       if os.path.isdir(wli_folder) and os.path.isdir(nbi_folder):
#         wli_images = sorted([os.path.join(wli_folder, f) for f in os.listdir(wli_folder) if f.endswith('.png')])
#         nbi_images = sorted([os.path.join(nbi_folder, f) for f in os.listdir(nbi_folder) if f.endswith('.png')])
#
#         # Assume that the corresponding images have the same file names in both folders
#         for wli_image, nbi_image in zip(wli_images, nbi_images):
#           image_pairs.append((wli_image, nbi_image))
#
#     return image_pairs

class CustomDataset(Dataset):
  def __init__(self, data_dir, class_name=None):
    self.data_dir = data_dir
    self.transform_WLI = transforms.Compose([
      transforms.Resize((384, 384)),
      transforms.ToTensor()
    ])
    self.transform_NBI = transforms.Compose([
      transforms.Resize((384, 384)),
      transforms.ToTensor()
    ])
    self.c_name = class_name
    self.class_to_idx = {c_name: idx for idx, c_name in enumerate(self.c_name.split(","))}
    self.image_pairs = self._get_image_pairs()

  def __len__(self):
    return len(self.image_pairs)

  def __getitem__(self, index):
    wli_path, nbi_path, target = self.image_pairs[index]
    wli_image = Image.open(wli_path).convert('RGB')
    nbi_image = Image.open(nbi_path).convert('RGB')

    if self.transform_WLI:
      wli_image = self.transform_WLI(wli_image)
    if self.transform_NBI:
      nbi_image = self.transform_NBI(nbi_image)

    return wli_image, nbi_image, index, target

  def _get_image_pairs(self):
    image_pairs = []
    folder_name = [s.strip() for s in self.c_name.split(",")]
    for category in folder_name:
      wli_folder = os.path.join(self.data_dir, f"{category}_WLI")
      nbi_folder = os.path.join(self.data_dir, f"{category}_NBI")
      target = self.class_to_idx[category]

      if os.path.isdir(wli_folder) and os.path.isdir(nbi_folder):
        wli_images = sorted([os.path.join(wli_folder, f) for f in os.listdir(wli_folder) if f.endswith('.png')])
        nbi_images = sorted([os.path.join(nbi_folder, f) for f in os.listdir(nbi_folder) if f.endswith('.png')])

        # Assume that the corresponding images have the same file names in both folders
        for wli_image, nbi_image in zip(wli_images, nbi_images):
          image_pairs.append((wli_image, nbi_image, target))

    return image_pairs
