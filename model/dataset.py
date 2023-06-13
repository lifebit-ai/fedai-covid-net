import os
import random
from math import ceil

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, sampler


class CustomDataset(Dataset):
    def __init__(self, root_dir: str, **kwargs) -> None:
        """
        Constructor for CovidCTDataset class.

        Args:
            root_dir (string): Directory with all the images.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir

        txt_COVID = kwargs.get("txt_COVID", "")
        txt_NonCOVID = kwargs.get("txt_NonCOVID", "")
        mode = kwargs.get("mode", "train")

        self.txt_path = [txt_COVID, txt_NonCOVID]
        self.classes = ["CT_COVID", "CT_NonCOVID"]
        self.num_cls = len(self.classes)
        self.img_list = []
        self.full_volume = None
        self.affine = None
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir, self.classes[c], item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transformer = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        val_transformer = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
        if mode == "train":
            self.transform = train_transformer

        else:
            self.transform = val_transformer
        print("samples = ", len(self.img_list))

    def __len__(self) -> int:
        """
        Returns the total number of samples.
        """
        return len(self.img_list)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns the sample with index `idx`.

        Args:
            idx (int): Index of the sample
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(int(self.img_list[idx][1]), dtype=torch.long)


def read_txt(txt_path: str) -> str:
    """
    Read the text file and return the data

    Args:
        txt_path: Path to the text file

    Returns: Data from the text file
    """
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


def get_random_sample_from_dataloader(dataloader: DataLoader, n: int, batch_size: int = 4) -> DataLoader:
    """
    Returns a random sample of size n from the dataloader

    Args:
        dataloader: Dataloader from which the sample is to be taken
        n: Size of the sample

    Returns: Random sample of size n from the dataloader
    """
    dataset = dataloader.dataset
    indices = list(range(len(dataset)))
    random_indices = random.sample(indices, n)
    random_sampler = sampler.SubsetRandomSampler(random_indices)
    random_dataloader = DataLoader(dataset, sampler=random_sampler, batch_size=batch_size)
    return random_dataloader


# Create a load_data function that returns trainloader, testloader, and num_examples
def load_data(batch_size: int = 4, root_dir: str = "../data/covid-ct/Images-processed/", **kwargs):
    """
    Loads the data and returns trainloader, testloader, and num_examples.

    Args:
        batch_size (int): Batch size for the data loaders
        root_dir (str): Path to the root directory containing the images

    Returns:
        trainloader (torch.utils.data.DataLoader): Data loader for the training set
        testloader (torch.utils.data.DataLoader): Data loader for the test set
        num_examples (dict): Dictionary containing the number of examples in the train and test sets
    """
    splits = ["train", "test", "val"]

    covid_datasets = {}
    txt_covid_dir = kwargs.get("txt_covid_dir", "../data/covid-ct/Data-split/COVID/")
    txt_non_covid_dir = kwargs.get("txt_non_covid_dir", "../data/covid-ct/Data-split/NonCOVID/")
    local_train = kwargs.get("local_train", False)

    for split in splits:
        covid_datasets[split] = CustomDataset(
            mode=split,
            root_dir=root_dir,
            txt_COVID=f"{txt_covid_dir}{split}CT_COVID.txt",
            txt_NonCOVID=f"{txt_non_covid_dir}{split}CT_NonCOVID.txt",
            transform=None,
        )

    # Create data generators
    # Note that we are only using the test and train sets and not the validation set
    trainloader = DataLoader(covid_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(covid_datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0)

    if not local_train:
        # Get a random subset of the trainloader for 3 nodes by splitting the trainloader into 3
        trainloader = get_random_sample_from_dataloader(
            trainloader, n=ceil(len(trainloader) / 3), batch_size=batch_size
        )

    num_examples = {
        "trainset": len(covid_datasets["train"]),
        "testset": len(covid_datasets["test"]),
    }

    return trainloader, testloader, num_examples
