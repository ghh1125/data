import torch
import random
import xlrd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
import math


seed=2481
random.seed(seed)



def ConvertELogStrToValue(eLogStr):
    """
    convert string of natural logarithm base of E to value
    return (convertOK, convertedValue)
    eg:
    input:  -1.1694737e-03
    output: -0.001169
    input:  8.9455025e-04
    output: 0.000895
    """
    foundEPower = re.search("(?P<coefficientPart>-?\d+\.\d+)e(?P<ePowerPart>-\d+)", eLogStr, re.I)

    if (foundEPower):
        coefficientPart = foundEPower.group("coefficientPart")
        ePowerPart = foundEPower.group("ePowerPart")

        coefficientValue = float(coefficientPart)
        ePowerValue = float(ePowerPart)

        wholeOrigValue = coefficientValue * math.pow(10, ePowerValue)


        (convertOK, convertedValue) = (True, wholeOrigValue)
    else:
        (convertOK, convertedValue) = (False, 0.0)

    return (convertOK, convertedValue)






class AppDataset(Dataset):
    def __init__(self, data_folder, train):
        super(AppDataset, self).__init__()
        self.is_train=train


        if data_folder=='BRCA' or data_folder=='ROSMAP':
            print('preparing BRCA data ...............')
            self.num_modality = 3

            if self.is_train:
                labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
                self.labels_tr = labels_tr.astype(int)
                self.labels_tr = torch.LongTensor(self.labels_tr)
                self.length = len(self.labels_tr)

                data_tr_list = []

                for i in range(1, self.num_modality + 1):
                    data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=','))

                # important
                # preprocessing. Preprocessing and feature preselection were performed on each omics data type individually to remove noise,
                # artifacts, and redundant features that may deteriorate the performance of the classification tasks.
                eps = 1e-10
                X_train_min_tr = [np.min(data_tr_list[i], axis=0, keepdims=True) for i in range(len(data_tr_list))]
                data_tr_list = [data_tr_list[i] - np.tile(X_train_min_tr[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
                X_train_max_tr = [np.max(data_tr_list[i], axis=0, keepdims=True) + eps for i in range(len(data_tr_list))]
                data_tr_list = [data_tr_list[i] / np.tile(X_train_max_tr[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]

                self.data_train_list = []
                for i in range(self.num_modality):
                    self.data_train_list.append(torch.FloatTensor(data_tr_list[i]))

                self.data = self.data_train_list
                self.label = self.labels_tr

            else:
                labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
                self.labels_te = labels_te.astype(int)
                self.labels_te = torch.LongTensor(self.labels_te)
                self.length = len(self.labels_te)

                data_te_list = []

                for i in range(1, self.num_modality + 1):
                    data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=','))
                eps = 1e-10
                X_train_min_te = [np.min(data_te_list[i], axis=0, keepdims=True) for i in range(len(data_te_list))]
                data_te_list = [data_te_list[i] - np.tile(X_train_min_te[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_te_list))]
                X_train_max_te = [np.max(data_te_list[i], axis=0, keepdims=True) + eps for i in range(len(data_te_list))]
                data_te_list = [data_te_list[i] / np.tile(X_train_max_te[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_te_list))]

                self.data_test_list = []
                for i in range(self.num_modality):
                    self.data_test_list.append(torch.FloatTensor(data_te_list[i]))

                self.data = self.data_test_list
                self.label = self.labels_te


        elif data_folder=='TP':

            print('preparing TP data ...............')
            self.num_modality = 2



            if self.is_train:
                self.label = []
                self.text_list = []
                self.image_list = []

                dataset = xlrd.open_workbook('TP/math-tr.xls')

                table = dataset.sheet_by_index(sheetx=0)

                print('preparing text_feature ...............')
                # text_feature
                for i in range(1,table.nrows):
                    cellV = table.cell_value(rowx=i, colx=1)
                    cellV = cellV.split(',')
                    temp = []
                    for j in cellV:
                        temp.append(ConvertELogStrToValue(j)[1])
                    assert len(temp)==768
                    self.text_list.append(temp)
                    # if i % 2000==0:
                    #     print(i,'/17772')
                print('preparing image_feature ...............')
                # image_feature
                for i in range(1,table.nrows):
                    cellV = table.cell_value(rowx=i, colx=2)
                    cellV = cellV.split(',')
                    temp = []
                    for j in cellV:
                        temp.append(ConvertELogStrToValue(j)[1])
                        if len(temp) == 768:
                            break
                    assert len(temp) == 768
                    self.image_list.append(temp)
                    # if i % 2000==0:
                    #     print(i,'/17772')

                # label
                for i in range(1,table.nrows):
                    cellV = table.cell_value(rowx=i, colx=3)
                    convert = {'A': 0, 'B': 1, 'C': 2}
                    self.label.append(convert[cellV])
                    # if i % 2000==0:
                    #     print(i,'/17772')

                self.label = torch.LongTensor(self.label)
                self.length = len(self.label)

                self.data = []

                self.data.append(torch.FloatTensor(self.text_list))  # [[text],[image]]
                self.data.append(torch.FloatTensor(self.image_list))

            else:
                self.label = []
                self.text_list = []
                self.image_list = []

                dataset = xlrd.open_workbook('TP/math-te.xls')

                table = dataset.sheet_by_index(sheetx=0)

                print('preparing text_feature ...............')
                # text_feature
                for i in range(1,table.nrows):
                    cellV = table.cell_value(rowx=i, colx=1)
                    cellV = cellV.split(',')
                    temp = []
                    for j in cellV:
                        temp.append(ConvertELogStrToValue(j)[1])
                    assert len(temp) == 768
                    self.text_list.append(temp)
                    # if i % 2000==0:
                    #     print(i,'/17772')
                print('preparing image_feature ...............')
                # image_feature
                for i in range(1,table.nrows):
                    cellV = table.cell_value(rowx=i, colx=2)
                    cellV = cellV.split(',')
                    temp = []
                    for j in cellV:
                        temp.append(ConvertELogStrToValue(j)[1])
                        if len(temp) == 768:
                            break
                    assert len(temp) == 768
                    self.image_list.append(temp)
                    # if i % 2000==0:
                    #     print(i,'/17772')

                # label
                for i in range(1,table.nrows):
                    cellV = table.cell_value(rowx=i, colx=3)
                    convert = {'A': 0, 'B': 1, 'C': 2}
                    self.label.append(convert[cellV])
                    # if i % 100==0:
                    #     print(i,'/1212')

                self.label = torch.LongTensor(self.label)
                self.length = len(self.label)

                self.data = []
                self.data.append(torch.FloatTensor(self.text_list))
                self.data.append(torch.FloatTensor(self.image_list))


    def __getitem__(self, item):
        data = []
        for i in range(self.num_modality):
            data.append(self.data[i][item])

        return self.label[item], data


    def __len__(self):
        return self.length




def collate_fn(batches):

    return 0


def get_train_data_loader(data_folder, batch_size=16):
    train_dataset = AppDataset(data_folder, train=True)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    return train_data_loader

def get_test_data_loader(data_folder, batch_size=16):
    test_dataset = AppDataset(data_folder, train=False)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return test_data_loader




