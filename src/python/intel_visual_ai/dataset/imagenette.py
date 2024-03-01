from intel_visual_ai.dataset.dataset_base import DatasetBase
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from torchvision import transforms
import json
import os


class ImageNette(DatasetBase):
    """
    Dataset handler for Imagenette dataset.

    Inherits from DatasetBase to use common dataset functionalities for downloading,
    extracting, and preparing the Imagenette dataset for use in processes like quantization.

    Attributes:
        quantization_batch_size (int): Batch size for quantization.
        dataset_path (str): Local path for storing the dataset.
        dataset_url (str): URL for downloading the Imagenette dataset.
        dataset_tar_filename (str): Name of the tar file to download for the Imagenette dataset.
        class_index_url (str): URL for downloading the ImageNet class index (used for label mapping).
    """

    def __init__(
        self,
        dataset_url: str = "https://s3.amazonaws.com/fast-ai-imageclas/",
        quantization_batch_size: int = 64,
        dataset_path: str = None,
        dataset_tar_filename: str = "imagenette2-320.tgz",
        class_index_url: str = "https://storage.googleapis.com/download.tensorflow.org/data/",
    ):
        """
        Initializes the Imagenette dataset handler with the provided parameters.

        Args:
            dataset_url (str, optional): URL for downloading the Imagenette dataset. Defaults to 'https://s3.amazonaws.com/fast-ai-imageclas/'.
            quantization_batch_size (int, optional): Batch size for quantization. Defaults to 64.
            dataset_path (str, optional): Custom path to store the dataset. Defaults to a subfolder in the current directory.
            dataset_tar_filename (str, optional): Name of the tar file for the Imagenette dataset. Defaults to 'imagenette2-320.tgz'.
            class_index_url (str, optional): URL path to class index of ImageNet class index for mapping. Defaults to 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'.
        """
        dataset_name = "imagenette"
        self.class_index_url = class_index_url
        super().__init__(
            dataset_url=dataset_url,
            dataset_name=dataset_name,
            dataset_tar_filename=dataset_tar_filename,
            dataset_label_filename="imagenet_class_index.json",
            dataset_label_url=class_index_url,
            dataset_path=dataset_path,
            quantization_batch_size=quantization_batch_size,
            no_proxy=None,
        )

    def download_class_index(self, label_file_path):
        """
        Downloads class index for ImageNette mapping to ImageNet class index.

        Args:
            label_file_path: path to json file
        """
        with open(label_file_path) as labels_file:
            class_index = json.load(labels_file)
        # Convert the class index to the desired format: {class_id: numeric_label}
        class_to_label = {v[0]: int(k) for k, v in class_index.items()}
        return class_to_label

    class ImageDataset(Dataset):
        """
        Custom dataset class for ImageNet.
        """

        def __init__(self, image_dir, class_to_label, transform=None):
            self.image_dir = image_dir
            self.transform = transform
            self.image_filenames = []
            self.labels = []

            # Read image paths and their corresponding labels
            for root, _, files in os.walk(image_dir):
                for file in files:
                    if file.endswith(".JPEG"):
                        class_id = os.path.basename(root)
                        label = class_to_label.get(
                            class_id, -1
                        )  # Default to -1 if class_id not found
                        self.image_filenames.append(os.path.join(root, file))
                        self.labels.append(label)

        def __len__(self):
            return len(self.image_filenames)

        def __getitem__(self, index):
            img_path = self.image_filenames[index]
            label = self.labels[index]
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label)

    def get_dataset(self, transform=None) -> Dataset:
        """
        Returns the ImageNette dataset.

        Args:
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, `transforms.Compose`

        Returns:
            Dataset: The ImageNette dataset.
        """
        images_folder, label_file_path = self.download_and_extract_files()
        class_to_label = self.download_class_index(label_file_path)
        # self.download_and_extract_files() returns two image and label, so only take the image
        return self.ImageDataset(images_folder[0], class_to_label, transform)

    def prepare_calibration_dataloader(self, device: str = "cpu") -> DataLoader:
        """
        Prepares the DataLoader for the ImageNette dataset.

        Args:
            device (str): The device to load the dataset onto (e.g., 'cpu', 'xpu').

        Returns:
            DataLoader: DataLoader with the ImageNette calibration dataset.
        """
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        dataset = self.get_dataset(transform)
        return DataLoader(dataset, batch_size=self.quantization_batch_size, shuffle=False)
