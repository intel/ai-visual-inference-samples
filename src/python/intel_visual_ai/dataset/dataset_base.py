import os
import requests
import tarfile
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


# Configure logging
logger = logging.getLogger("dataset")
logger.addHandler(TqdmLoggingHandler())


class DatasetBase:
    """
    Base class for dataset handlers.

    This class provides functionalities to download and extract dataset files. It serves
    as a base for dataset-specific handler classes.

    Attributes:
        dataset_url (str): URL of the dataset.
        _dataset_tar_filename (str): Name of the dataset tar file.
        _dataset_label_filename (str): Name of the dataset label file.
        quantization_batch_size (int): Batch size for quantization.
        dataset_path (str): Local path for storing the dataset.
        _tar_file_path (str): Full path to the dataset tar file.
        _label_file_path (str): Full path to the dataset label file.
        _images_folder (str): Directory to store extracted images.
    """

    def __init__(
        self,
        dataset_url: str,
        dataset_name: str,
        dataset_tar_filename: str,
        dataset_label_filename: str,
        dataset_label_url: str,
        dataset_path: str,
        quantization_batch_size: int,
        no_proxy: str = None,
    ):
        """
        Initializes the DatasetBase class with provided parameters.

        Args:
            dataset_url (str): URL of the dataset.
            dataset_name (str): Name of the dataset.
            dataset_tar_filename (str): Name of the dataset tar file.
            dataset_label_filename (str): Name of the dataset label file.
            dataset_path (str): Local path for storing the dataset.
            quantization_batch_size (int): Batch size for quantization.
            no_proxy (str): Exclude URL from proxy server for URL download.
        """
        self.dataset_url = dataset_url
        self._dataset_tar_filename = dataset_tar_filename
        self.quantization_batch_size = quantization_batch_size
        self.dataset_path = dataset_path or os.path.join(os.path.dirname(__file__), dataset_name)
        self.dataset_label_url = dataset_label_url
        if no_proxy:
            os.environ["no_proxy"] = no_proxy
            logger.info(f"Proxy exclusion set for: {no_proxy}")

        # Initialize full paths for the dataset tar file and labels file
        self._tar_file_path = os.path.join(self.dataset_path, self._dataset_tar_filename)
        self._dataset_label_filename = None
        if dataset_label_filename:
            self._dataset_label_filename = dataset_label_filename
            self._label_file_path = os.path.join(self.dataset_path, self._dataset_label_filename)
        self._images_folder = os.path.join(self.dataset_path, "images")

    def _download_file(self, url: str, filename: str) -> None:
        """
        Downloads a file from the specified URL.

        Args:
            url (str): URL to download the file from.
            filename (str): Name of the file to save the download as.
        """
        response = requests.get(url, stream=True, timeout=30)
        file_size = int(response.headers.get("Content-Length", 0))
        with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as progress:
            with open(filename, "wb") as f:
                for data in response.iter_content(1024):
                    f.write(data)
                    progress.update(len(data))
            logger.info(f"Downloading {filename} completed.")

    def _extract_tarfile(self, filename: str, path: str) -> None:
        """
        Extracts a tar file to the specified path.

        Args:
            filename (str): Tar file to extract.
            path (str): Directory to extract the tar file to.
        """
        with tarfile.open(filename) as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), unit="files") as progress:
                for member in members:
                    tar.extract(member, path=path)
                    progress.update(1)
            logger.info(f"Extracting {filename} completed.")

    def download_and_extract_files(self) -> tuple[str, str]:
        """
        Downloads and extracts dataset files.

        This method checks for the existence of the dataset files,
        downloads them if not present, and then extracts them.

        Returns:
            tuple[str, str]: A tuple containing paths to the images folder and label file.
        """
        if not os.path.exists(self._tar_file_path):
            os.makedirs(self.dataset_path, exist_ok=True)
            logger.info("Creating dataset directory and downloading tar file.")
            self._download_file(self.dataset_url + self._dataset_tar_filename, self._tar_file_path)
            self._extract_tarfile(self._tar_file_path, self._images_folder)
        else:
            logger.info("Dataset tarball already exists, skipping downloading.")

        if self._dataset_label_filename and not os.path.exists(self._label_file_path):
            logger.info("Downloading label file.")
            self._download_file(
                self.dataset_label_url + self._dataset_label_filename, self._label_file_path
            )
        else:
            logger.info(
                "Labels file already exists or Label URL was not given to dataset, skipping downloading."
            )

        return self._images_folder, self._label_file_path if self._dataset_label_filename else None
