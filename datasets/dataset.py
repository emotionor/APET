from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import io
import torch

# from datasets.config import Years

class Dos_Dataset(Dataset):
    def __init__(self, data_dir="./data", split='train', smear=0, choice=[],**kwargs) -> None:
        super().__init__()
        self.split = split
        self.smear = smear
        self.data_dir = data_dir+"/"+split+"/"
        self.src_len   = 98  # elements length 
        self.tgt_len   = 40  # target dos length
        
        self.elements  = self.get_elements()  #size (__len__, src_len)
        self.positions = self.get_positions() #size (__len__, src_len*3)
        self.tgtdos    = self.get_tgtdos()    #size (__len__, tge_len)

        self.choice = choice
        if self.choice:
            cholist = torch.Tensor(self.choice).int()
            self.elements  = self.elements.index_select(dim=0, index=cholist)
            self.positions = self.positions.index_select(dim=0, index=cholist)
            self.tgtdos    = self.tgtdos.index_select(dim=0, index=cholist)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, index):
        # type tensor; size [src_len, src_len*3, tgt_len]
        index = min(index, self.__len__())
        array_seq = [self.elements[index], self.positions[index].reshape(-1, 3), self.tgtdos[index]] 
        return array_seq

    def get_elements(self):
        if self.smear==0:
            filename  = self.data_dir+"elements_%s_new.npy"%self.split
        else:
            filename  = self.data_dir+"elements_g%s_%s.npy"%(self.smear, self.split)
        elements  = np.load(filename)
        return torch.Tensor(elements).long()

    def get_positions(self):
        if self.smear==0:
            filename  = self.data_dir+"positions_%s_new.npy"%self.split
        else:
            filename  = self.data_dir+"positions_g%s_%s.npy"%(self.smear, self.split)
        positions  = np.load(filename)
        return torch.Tensor(positions)

    def get_tgtdos(self):
        if self.smear==0:
            filename  = self.data_dir+"tgtdos_%s.npy"%self.split
        else:
            filename  = self.data_dir+"tgtdos_g%s_%s.npy"%(self.smear, self.split)
        tgtdos  = np.load(filename)
        return torch.Tensor(tgtdos)

if __name__ == "__main__":
    test = Dos_Dataset(data_dir="../data", split="test")
    print(test.__getitem__(15))
