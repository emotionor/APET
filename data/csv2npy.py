import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split


def csv2npy(df_all, pwd, split, stratify=None, random_states=0):
    train_df, remaining_df = train_test_split(df_all, test_size=0.2, random_state=random_states, stratify=stratify)
    valid_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=random_states, stratify=stratify)
    np.save(os.path.join(pwd, "train","%s_apet_train.npy"%split), train_df.to_numpy())
    np.save(os.path.join(pwd, "valid","%s_apet_valid.npy"%split), valid_df.to_numpy())
    np.save(os.path.join(pwd, "test","%s_apet_test.npy"%split),   test_df.to_numpy())

# Please change the APET PATH
pwd = "./train4w2023"

def mkdirdt(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
mkdirdt(os.path.join(pwd,"train"))
mkdirdt(os.path.join(pwd,"valid"))
mkdirdt(os.path.join(pwd,"test"))
src_len = 108
tgt_len = 128

filename1 = "elements_apet.csv"
filename2 = "positions_apet.csv"
filename3  = "tgtdos_apet.csv"



names1     = [str(i) for i in range(src_len+1)]
elements  = pd.read_csv(filename1, names=names1, header=None, index_col=0,).fillna(0)
csv2npy(elements, pwd, "elements")

names2     = [str(i) for i in range(src_len*3+1)]
positions = pd.read_csv(filename2, names=names2, header=None, index_col=0,).fillna(0)
csv2npy(positions, pwd, "positions")

names3     = [str(i) for i in range(tgt_len+1)]
dos_data  = pd.read_csv(filename3, names=names3, header=None, index_col=0,)
csv2npy(dos_data, pwd, "tgtdos")


