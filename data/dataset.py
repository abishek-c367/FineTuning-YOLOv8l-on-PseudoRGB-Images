# Import the os module for file system operations
import os
# Import PyTorch for tensor operations
import torch
# Import Dataset class from torch.utils.data to create a custom dataset
from torch.utils.data import Dataset
# Import Image from PIL for image loading and manipulation
from PIL import Image
# Import torchvision.transforms for image transformations
import torchvision.transforms as transforms


# Define a custom dataset class for unpaired image-to-image translation
class UnpairedDataset(Dataset):
    # Initialize the dataset with root directories for domain A and B, and optional transforms
    def __init__(self, root_dir_A, root_dir_B, transform=None):
        # Store the root directory path for domain A images
        self.root_dir_A = root_dir_A
        # Store the root directory path for domain B images
        self.root_dir_B = root_dir_B
        # Store the transformation to apply to images
        self.transform = transform

        # Get a sorted list of filenames in domain A directory
        self.files_A = sorted(os.listdir(root_dir_A))
        # Get a sorted list of filenames in domain B directory
        self.files_B = sorted(os.listdir(root_dir_B))

        # For unpaired datasets, we can pair images in various ways; here we assume sequential pairing
        # Calculate the length of the file list for domain A
        self.len_A = len(self.files_A)
        # Calculate the length of the file list for domain B
        self.len_B = len(self.files_B)

    # Return the length of the dataset, using the maximum length of the two domains
    def __len__(self):
        return max(self.len_A, self.len_B)

    # Get an item from the dataset at the given index
    def __getitem__(self, idx):
        # Construct the path to the image in domain A, using modulo to cycle through if necessary
        img_A_path = os.path.join(self.root_dir_A, self.files_A[idx % self.len_A])
        # Construct the path to the image in domain B, using modulo to cycle through if necessary
        img_B_path = os.path.join(self.root_dir_B, self.files_B[idx % self.len_B])

        # Open the image from domain A and convert to RGB
        img_A = Image.open(img_A_path).convert('RGB')
        # Open the image from domain B and convert to RGB
        img_B = Image.open(img_B_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            # Transform image A
            img_A = self.transform(img_A)
            # Transform image B
            img_B = self.transform(img_B)

        # Return a dictionary with the transformed images from both domains
        return {'A': img_A, 'B': img_B}
if __name__ == "__main__":
    # Example usage of the UnpairedDataset with the correct FLIR ADAS paths
    dataset = UnpairedDataset(
        'data/flir_adas/images_rgb_train/data',
        'data/flir_adas/images_thermal_train/data',
        transform=transforms.ToTensor()
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[10742]
    print(f"Data type of sample: {type(sample)}")
    print(f"Sample keys: {sample.keys()}")
    print(f"Sample A shape: {sample['A'].shape}, Sample B shape: {sample['B'].shape}")