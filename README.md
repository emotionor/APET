# APET
[Atomic Positional Embedding-based Transformer introduced by Cui Yaning, et al.](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.3c02036)

## Dataset
The DOS dataset are avaliable in Material Project (https://next-gen.materialsproject.org/).
All the data this work uses have been converted to three csv file, which is visible in the ```./data```. 
The data files contain the corresponding MP-ids in Material Project, which can be verified by all by themselves.

## Run the model
Run ```./data/csv2npy.py``` to generate the dataset for the APET pretraining task.
Run ```train_script.sh``` to start running the APET pretraining task.
