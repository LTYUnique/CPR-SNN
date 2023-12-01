from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np


def load_train(args):
    # get ids 
    with open(f"{args.root}{args.data_path}/{args.train_list}.txt", "r") as f:  #打开文本  #, encoding='utf-8'
        train_list = f.read()   #读取文本
    train_list = train_list.split('\n')

    
    if args.data == 'ISPY1':
        train_set = DataGen(args.root, args.data_path, train_list)
    elif args.data == 'pilot':
        train_set = DataGen_pilot(args.root, args.data_path, train_list)
    else:
        raise RuntimeError('error data name', args.data)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)

    return train_loader

def load_test(args):
    # get ids 
    with open(f"{args.root}{args.data_path}/{args.test_list}.txt", "r") as f:  #打开文本  #, encoding='utf-8'
        test_list = f.read()   #读取文本
    test_list = test_list.split('\n')
    
    if args.data == 'ISPY1':
        test_set = TestGen(args.root, args.data_path, test_list)
    elif args.data == 'pilot':
        test_set = TestGen_pilot(args.root, args.data_path, test_list, args.zero_prompt_test, args.n_clin_var)
    else:
        raise RuntimeError('error data name', args.data)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    return DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)


class DataGen(Dataset):
    def __init__(self, root, data_path, name_list):
        self.root = root
        self.data_path = data_path
        self.name_list = name_list

    def __getitem__(self, index):
        data = np.load(f'{self.root}{self.data_path}/{self.name_list[index]}.npy', allow_pickle=True)
        return data[0], data[1], data[2], data[3], data[4]  # [img1, img2, seg, pcr, prompt]
    
    def __len__(self):
        return len(self.name_list)

class DataGen_pilot(Dataset):
    def __init__(self, root, data_path, name_list):
        self.root = root
        self.data_path = data_path
        self.name_list = name_list

    def __getitem__(self, index):
        data = np.load(f'{self.root}{self.data_path}/{self.name_list[index]}', allow_pickle=True)
        return data[0], data[1], data[2], data[3], data[4]  # [img1, img2, seg, pcr, prompt]
    
    def __len__(self):
        return len(self.name_list)
    
class TestGen(Dataset):
    def __init__(self, root, data_path, name_list):
        self.root = root
        self.data_path = data_path
        self.name_list = name_list

    def __getitem__(self, index):
        data = np.load(f'{self.root}{self.data_path}/{self.name_list[index]}.npy', allow_pickle=True)
        return self.name_list[index], data[0], data[1], data[2], data[3], data[4]
    
    def __len__(self):
        return len(self.name_list)

class TestGen_pilot(Dataset):
    def __init__(self, root, data_path, name_list, zero_prompt, n_clin_var):
        self.root = root
        self.data_path = data_path
        self.name_list = name_list
        self.zero_prompt = zero_prompt
        self.n_clin_var = n_clin_var

    def __getitem__(self, index):
        data = np.load(f'{self.root}{self.data_path}/{self.name_list[index]}', allow_pickle=True)
        prompt = np.zeros(self.n_clin_var) if self.zero_prompt else data[4]
        return self.name_list[index], data[0], data[1], data[2], data[3], prompt
    
    def __len__(self):
        return len(self.name_list)
    
