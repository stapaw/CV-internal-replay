from torch.utils.data import Dataset
import torchvision
import os
from google_drive_downloader import GoogleDriveDownloader as gdd


class fruits360(Dataset):
    def __init__(self, root_dir, train=True, download=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        if train:
            self.dir = os.path.join(self.root_dir, "train")
        else:
            self.dir = os.path.join(self.root_dir, "test")
        self.transforms = transform
        self.target_transform = target_transform
        if download:
            self.download(path=self.root_dir)

        self.dataset = torchvision.datasets.ImageFolder(
            root=self.dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if self.target_transform:
            target = self.target_transform(target)
        return (img, target)

    def download(self, path):
        '''
        Download dataset if not present
        '''
        url = "https://drive.google.com/file/d/1h96fQdKVII3nWvvrrO5KXIZ8msZSJ6wa/view?usp=sharing"
        id = url.split("/")[5]
        zipName = "fruits360.zip"
        if not os.path.isdir(path):
            ogPath = os.path.split(path)[0]
            gdd.download_file_from_google_drive(file_id=id,
                                                dest_path=os.path.join(
                                                    ogPath, zipName),
                                                unzip=True)
            os.remove(os.path.join(ogPath, zipName))
            print("Downloaded")
        else:
            print("Dataset already downloaded")