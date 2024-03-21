from datasets.FER_dataset import FERDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

def FERDataloader(data_dir, val_size=0.2, batch_size=32, shuffle=True, transformations=None): # TODO: change into a class
    '''This is function to load the dataset and returns the dataloaders
        Args:
            data_dir: path to the dataset
            batch_size: batch size for the dataloader
            num_workers: number of workers for the dataloader
            shuffle: whether to shuffle the dataset or not
            debug: whether to run in debug mode or not
        Returns:
            train_loader: dataloader for the training set
            val_loader: dataloader for the validation set
            test_loader: dataloader for the test set
    '''
    train_dataset = FERDataset(
        data_dir=data_dir, split='train', transform=transformations)
    test_dataset = FERDataset(
        data_dir=data_dir, split='test', transform=transformations)


    val_len = int(val_size*len(train_dataset))
    train_ds, val_ds = random_split(
        train_dataset, [len(train_dataset)-val_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader
