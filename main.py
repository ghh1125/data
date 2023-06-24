from torch import true_divide
from train_test import train

if __name__ == "__main__":
    datafolder = 'TP'
    #datafolder = 'BRCA'
    #datafolder = 'ROSMAP'
    train(datafolder)