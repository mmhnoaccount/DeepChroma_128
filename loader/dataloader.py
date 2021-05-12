# import pickle, random
# import math, time, platform
import random
import time, platform
from tfrecord.torch.dataset import TFRecordDataset
# rom pathlib import Path

# import cv2
import torch
import numpy as np
# from PIL import Image
# from skimage import color

# from torchvision import transforms, datasets
# from torchvision.transforms import functional as tvF
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset  # For custom datasets

import os
from temp_visualize import sort_humanly

import tfrecord
from tfrecord.torch.dataset import TFRecordDataset


def set_seed(seed, print_log=True):
    if seed < 0:
        return
    if print_log:
        print('set random seed: {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


'''
TrainSet_Pack = 'F:/train_set'
Train_Cb_org = TrainSet_Pack + '/Cb_32_32/org'
Train_Cr_org = TrainSet_Pack + '/Cr_32_32/org'
Train_Luma = TrainSet_Pack + '/luma_32_32/CCLM'
Train_Cb_CCLM = TrainSet_Pack + '/Cb_32_32/CCLM'
Train_Cr_CCLM = TrainSet_Pack + '/Cr_32_32/CCLM'
Train_QP = TrainSet_Pack + '/QP_32_32'

Train_Cb_DM = TrainSet_Pack + '/Cb_32_32/DM'
Train_Cr_DM = TrainSet_Pack + '/Cr_32_32/DM'
Train_Cb_Ref = TrainSet_Pack + '/Cb_32_32/Ref'
Train_Cr_Ref = TrainSet_Pack + '/Cr_32_32/Ref'


TestSet_Pack = 'F:/test_set'
Test_Cb_org = TestSet_Pack + '/Cb_32_32/org'
Test_Cr_org = TestSet_Pack + '/Cr_32_32/org'
Test_Luma = TestSet_Pack + '/luma_32_32/CCLM'
Test_Cb_CCLM = TestSet_Pack + '/Cb_32_32/CCLM'
Test_Cr_CCLM = TestSet_Pack + '/Cr_32_32/CCLM'
Test_QP = TestSet_Pack + '/QP_32_32'

Test_Cb_DM = TestSet_Pack + '/Cb_32_32/DM'
Test_Cr_DM = TestSet_Pack + '/Cr_32_32/DM'
Test_Cb_Ref = TestSet_Pack + '/Cb_32_32/Ref'
Test_Cr_Ref = TestSet_Pack + '/Cr_32_32/Ref'


VisualSet_pack = 'E:/visualize_set'
Visualize_Cb_org = VisualSet_pack + '/train_visualize_22/Cb_32_32/org'
Visualize_Cr_org = VisualSet_pack + '/train_visualize_22/Cr_32_32/org'
Visualize_Luma = VisualSet_pack + '/train_visualize_22/luma_32_32/CCLM'
Visualize_Cb_CCLM = VisualSet_pack + '/train_visualize_22/Cb_32_32/CCLM'
Visualize_Cr_CCLM = VisualSet_pack + '/train_visualize_22/Cr_32_32/CCLM'
Visualize_QP = VisualSet_pack + '/train_visualize_22/QP_32_32'

Visualize_Cb_DM = VisualSet_pack + '/train_visualize_22/Cb_32_32/DM'
Visualize_Cr_DM = VisualSet_pack + '/train_visualize_22/Cr_32_32/DM'
Visualize_Cb_Ref = VisualSet_pack + '/train_visualize_22/Cb_32_32/Ref'
Visualize_Cr_Ref = VisualSet_pack + '/train_visualize_22/Cr_32_32/Ref'
'''

TrainSet_Pack = 'E:/File/package_by_mmh/train_set_128'
Train_Cb_org = TrainSet_Pack + '/origCb'
Train_Cr_org = TrainSet_Pack + '/origCr'
Train_Cb_rec = TrainSet_Pack + '/RecCb'
Train_Cr_rec = TrainSet_Pack + '/RecCr'
Train_Luma = TrainSet_Pack + '/Luma'
Train_QP = TrainSet_Pack + '/QPmask'

Train_tfrecord = 'E:/File/package_by_mmh/train_set_128/train.tfrecord'


TestSet_Pack = 'F:/test_set'
Test_Cb_org = TestSet_Pack + '/origCb'
Test_Cr_org = TestSet_Pack + '/origCr'
Test_Cb_rec = TestSet_Pack + '/RecCb'
Test_Cr_rec = TestSet_Pack + '/RecCr'
Test_Luma = TestSet_Pack + '/Luma'
Test_QP = TestSet_Pack + '/QPmask'


#VisualSet_pack = 'E:/File/package_by_mmh/train_set_128/visualize_valid'
VisualSet_pack = r'E:\File\package_by_mmh\train_set_128\visualize_valid\trainVisualize_set'
Visualize_Cb_org = VisualSet_pack + '/origCb'
Visualize_Cr_org = VisualSet_pack + '/origCr'
Visualize_Cb_rec = VisualSet_pack + '/RecCb'
Visualize_Cr_rec = VisualSet_pack + '/RecCr'
Visualize_Luma = VisualSet_pack + '/Luma'
Visualize_QP = VisualSet_pack + '/QPmask'



def readMatrix(path, height, width, QP_mask=False):
    A = np.zeros((height, width), dtype=float)  # 先创建一个 3x3的全零方阵A，并且数据的类型设置为float浮点型

    f = open(path)  # 打开数据文件文件
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    A_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        per_line = line.strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        # print(list)
        A[A_row, :] = per_line[:width]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
        A_row += 1  # 然后方阵A的下一行接着读
        # print(line)
    if QP_mask:
        A = (A-(63.0/2.0)) / (63.0/2.0)
    else:
        A = (A-(1023.0/2.0)) / (1023.0/2.0)
    return A

'''
def readRef(path):
    height = 32
    width = 32
    A = np.zeros((1, height + width), dtype=float)
    f = open(path)  # 打开数据文件文件
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    line = lines[0]
    per_line = line.strip('\n').split(' ')
    #print(per_line)
    #print(per_line.shape)
    A[0, :] = per_line[:-1]
    A = (A-(1023.0/2.0)) / (1023.0/2.0)
    return A
'''

def catLuma(luma_path, index, Train, Visualize=False):
    if Visualize:
        Luma_Matrix = readMatrix(Visualize_Luma + '/' + luma_path[index], 128, 128)
    else:
        if Train:
            Luma_Matrix = readMatrix(Train_Luma + '/' + luma_path[index], 128, 128)
        else:
            Luma_Matrix = readMatrix(Test_Luma + '/' + luma_path[index], 128, 128)
    return np.expand_dims(Luma_Matrix, 0)

def catQP(qp_path, index, Train, Visualize=False):
    if Visualize:
        QP_Matrix = readMatrix(Visualize_QP + '/' + qp_path[index], 32, 32, QP_mask=True)
    else:
        if Train:
            QP_Matrix = readMatrix(Train_QP + '/' + qp_path[index], 32, 32, QP_mask=True)
        else:
            QP_Matrix = readMatrix(Test_QP + '/' + qp_path[index], 32, 32, QP_mask=True)
    return np.expand_dims(QP_Matrix, 0)

def catChroma(chroma_path, index, Train, Visualize=False):
    if Visualize:
        Cb_Matrix = readMatrix(Visualize_Cb_rec + '/' + chroma_path[0][index], 64, 64)
        Cr_Matrix = readMatrix(Visualize_Cr_rec + '/' + chroma_path[1][index], 64, 64)
        # QP_Matrix = readMatrix(Visualize_QP + '/' + chroma_path[2][index], QP_mask=True)
    else:
        if Train:
            Cb_Matrix = readMatrix(Train_Cb_rec + '/' + chroma_path[0][index], 64, 64)
            Cr_Matrix = readMatrix(Train_Cr_rec + '/' + chroma_path[1][index], 64, 64)
            # QP_Matrix = readMatrix(Visualize_QP + '/' + chroma_path[2][index], QP_mask=True)
        else:
            Cb_Matrix = readMatrix(Test_Cb_rec + '/' + chroma_path[0][index], 64, 64)
            Cr_Matrix = readMatrix(Test_Cr_rec + '/' + chroma_path[1][index], 64, 64)
            # QP_Matrix = readMatrix(Visualize_QP + '/' + chroma_path[2][index], QP_mask=True)
    return np.concatenate((np.expand_dims(Cb_Matrix, 0), np.expand_dims(Cr_Matrix, 0)), 0)


def catOrg(org_path, index, Train, Visualize=False):
    if Visualize:
        Cb_org_Matrix = readMatrix(Visualize_Cb_org + '/' + org_path[0][index], 32, 32)
        Cr_org_Matrix = readMatrix(Visualize_Cr_org + '/' + org_path[1][index], 32, 32)
    else:
        if Train:
            Cb_org_Matrix = readMatrix(Train_Cb_org + '/' + org_path[0][index], 32, 32)
            Cr_org_Matrix = readMatrix(Train_Cr_org + '/' + org_path[1][index], 32, 32)
        else:
            Cb_org_Matrix = readMatrix(Test_Cb_org + '/' + org_path[0][index], 32, 32)
            Cr_org_Matrix = readMatrix(Test_Cr_org + '/' + org_path[1][index], 32, 32)
    return np.concatenate((np.expand_dims(Cb_org_Matrix, 0), np.expand_dims(Cr_org_Matrix, 0)), 0)

'''
def catRef(Ref_path, index, Train, Visualize=False):
    if Visualize:
        Cb_Ref_Matrix = readRef(Visualize_Cb_Ref + '/' + Ref_path[2][index])
        Cr_Ref_Matrix = readRef(Visualize_Cr_Ref + '/' + Ref_path[3][index])
    else:
        if Train:
            Cb_Ref_Matrix = readRef(Train_Cb_Ref + '/' + Ref_path[2][index])
            Cr_Ref_Matrix = readRef(Train_Cr_Ref + '/' + Ref_path[3][index])
        else:
            Cb_Ref_Matrix = readRef(Test_Cb_Ref + '/' + Ref_path[2][index])
            Cr_Ref_Matrix = readRef(Test_Cr_Ref + '/' + Ref_path[3][index])
    return np.concatenate((Cb_Ref_Matrix, Cr_Ref_Matrix), 1)
'''

class LumaAndChromaDataset(Dataset):
    def __init__(self, luma_path, chroma_path, qp_path, org_path, Train=False, seed=-1, **kwargs):

        self.luma_path = luma_path
        self.chroma_path = chroma_path
        self.qp_path = qp_path
        self.org_path = org_path
        self.Train = Train

        self.data_len = len(luma_path)

        self.idx_shuffle = list(range(self.data_len))

        random.seed(10)
        random.shuffle(self.idx_shuffle)
        random.seed(time.time() if seed < 0 else seed)

    def __getitem__(self, idx):
        index = self.idx_shuffle[idx]

        Luma_id = torch.FloatTensor(catLuma(self.luma_path, index, self.Train))
        Chorma_id = torch.FloatTensor(catChroma(self.chroma_path, index, self.Train))
        QP_id = torch.FloatTensor(catQP(self.qp_path, index, self.Train))
        # Ref_id = torch.FloatTensor(catRef(self.chroma_path, index, self.Train)).squeeze()

        org_id = torch.FloatTensor(catOrg(self.org_path, index, self.Train))

        return (Luma_id, Chorma_id, QP_id, org_id)

    def __len__(self):
        return self.data_len


class visualize_Dataset(Dataset):
    def __init__(self):
        Luma_list = sort_humanly(os.listdir(Visualize_Luma))
        QP_list = sort_humanly(os.listdir(Visualize_QP))

        Cb_Rec_list = sort_humanly(os.listdir(Visualize_Cb_rec))
        Cr_Rec_list = sort_humanly(os.listdir(Visualize_Cr_rec))

        Cb_org_list = sort_humanly(os.listdir(Visualize_Cb_org))
        Cr_org_list = sort_humanly(os.listdir(Visualize_Cr_org))

        # Train_len = len(Luma_list)

        Luma_path = Luma_list
        Chroma_path = (Cb_Rec_list, Cr_Rec_list)
        QP_path = QP_list
        org_path = (Cb_org_list, Cr_org_list)


        self.luma_path = Luma_path
        self.chroma_path = Chroma_path
        self.org_path = org_path
        self.qp_path = QP_path
        self.Train = False
        self.Visualize = True

        self.data_len = len(Luma_path)

    def __getitem__(self, idx):
        index = idx

        Luma_id = torch.FloatTensor(catLuma(self.luma_path, index, self.Train, self.Visualize))
        Chorma_id = torch.FloatTensor(catChroma(self.chroma_path, index, self.Train, self.Visualize))
        # Ref_id = torch.FloatTensor(catRef(self.chroma_path, index, self.Train, self.Visualize)).squeeze()
        QP_id = torch.FloatTensor(catQP(self.qp_path, index, self.Train, self.Visualize))

        org_id = torch.FloatTensor(catOrg(self.org_path, index, self.Train, self.Visualize))

        return (Luma_id, Chorma_id, QP_id, org_id)

    def __len__(self):
        return self.data_len



def get_train_dataset(args):
    set_seed(args.seed)

    #data_dir_path = Path(args.data_dir)

    batch_size = args.batch_size
    input_size = args.input_size

    Train_Luma_list = os.listdir(Train_Luma)
    Train_Cb_Rec_list = os.listdir(Train_Cb_rec)
    Train_Cr_Rec_list = os.listdir(Train_Cr_rec)
    Train_QP_list = os.listdir(Train_QP)

    Train_Cb_org_list = os.listdir(Train_Cb_org)
    Train_Cr_org_list = os.listdir(Train_Cr_org)

    Train_len = len(Train_Luma_list)

    Train_Luma_path = Train_Luma_list
    Train_Chroma_path = (Train_Cb_Rec_list, Train_Cr_Rec_list)
    Train_QP_path = Train_QP_list
    Train_org_path = (Train_Cb_org_list, Train_Cr_org_list)

    # Train set
    print('making train set...')

    if platform.system() == 'Windows':
        _init_fn = None
    else:
        _init_fn = lambda worker_id: set_seed(args.seed, print_log=False)

    train = LumaAndChromaDataset(luma_path=Train_Luma_path, chroma_path=Train_Chroma_path, qp_path=Train_QP_path,
        org_path=Train_org_path, Train=True, seed=args.seed)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=args.thread, worker_init_fn=_init_fn)

    '''
    # Test set
    print('making test set...')

    Test_Luma_list = os.listdir(Test_Luma)
    Test_Cb_CCLM_list = os.listdir(Test_Cb_CCLM)
    Test_Cr_CCLM_list = os.listdir(Test_Cb_CCLM)
    Test_QP_list = os.listdir(Test_QP)

    Test_Cb_DM_list = os.listdir(Test_Cb_DM)
    Test_Cr_DM_list = os.listdir(Test_Cr_DM)
    Test_Cb_Ref_list = os.listdir(Test_Cb_Ref)
    Test_Cr_Ref_list = os.listdir(Test_Cr_Ref)

    Test_Cb_org_list = os.listdir(Test_Cb_org)
    Test_Cr_org_list = os.listdir(Test_Cb_org)

    Test_len = len(Test_Luma_list)

    Test_Luma_path = (Test_Luma_list, Test_Cb_CCLM_list, Test_Cr_CCLM_list, Test_QP_list)
    Test_Chroma_path = (Test_Cb_DM_list, Test_Cr_DM_list, Test_Cb_Ref_list, Test_Cr_Ref_list)
    Test_org_path = (Test_Cb_org_list, Test_Cr_org_list)

    test = LumaAndChromaDataset(luma_path=Test_Luma_path, chroma_path=Test_Chroma_path,
                                 org_path=Test_org_path, Train=False)

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=args.thread,
                              worker_init_fn=_init_fn)

    return train_loader, test_loader
    '''
    return train_loader


def get_test_dataset(args):

    batch_size = args.batch_size
    input_size = args.input_size

    # Test set
    print('making test set...')

    Test_Luma_list = os.listdir(Test_Luma)
    Test_Cb_Rec_list = os.listdir(Test_Cb_rec)
    Test_Cr_Rec_list = os.listdir(Test_Cr_rec)
    Test_QP_list = os.listdir(Test_QP)

    Test_Cb_org_list = os.listdir(Test_Cb_org)
    Test_Cr_org_list = os.listdir(Test_Cr_org)

    Test_len = len(Test_Luma_list)

    Test_Luma_path = Test_Luma_list
    Test_Chroma_path = (Test_Cb_Rec_list, Test_Cr_Rec_list)
    Test_QP_path = Test_QP_list
    Test_org_path = (Test_Cb_org_list, Test_Cr_org_list)

    test = LumaAndChromaDataset(luma_path=Test_Luma_path, chroma_path=Test_Chroma_path, qp_path=Test_QP_path,
                                org_path=Test_org_path, Train=False)

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=args.thread)

    return test_loader



def decode_image(features):
    features["Luma_id"] = torch.FloatTensor(np.frombuffer(features['Luma_id']).reshape(1, 128, 128).copy())
    features["Chorma_id"] = torch.FloatTensor(np.frombuffer(features['Chorma_id']).reshape(2, 64, 64).copy())
    features["QP_id"] = torch.FloatTensor(np.frombuffer(features['QP_id']).reshape(1, 32, 32).copy())
    features["org_id"] = torch.FloatTensor(np.frombuffer(features['org_id']).reshape(2, 32, 32).copy())
    return features

def get_train_dataset_tfrecord(args):
    set_seed(args.seed)

    batch_size = args.batch_size
    input_size = args.input_size

    if platform.system() == 'Windows':
        _init_fn = None
    else:
        _init_fn = lambda worker_id: set_seed(args.seed, print_log=False)

    description = {'Luma_id': 'byte', 'Chorma_id': 'byte', 'QP_id': 'byte', 'org_id': 'byte'}

    train = TFRecordDataset(Train_tfrecord,
                            index_path=None,
                            description=description,
                            transform=decode_image,
                            shuffle_queue_size=16384)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=args.thread,
                                               worker_init_fn=_init_fn)
    return train_loader


def get_dataset(args):
    if args.test:
        return get_test_dataset(args)
    else:
        if args.use_tfrecord:
            return get_train_dataset_tfrecord(args)
        else:
            return get_train_dataset(args)


if __name__ == '__main__':
    batch_size = 1
    '''
    Test_Luma_list = os.listdir(Test_Luma)
    Test_Cb_Rec_list = os.listdir(Test_Cb_rec)
    Test_Cr_Rec_list = os.listdir(Test_Cb_rec)
    Test_QP_list = os.listdir(Test_QP)

    Test_Cb_org_list = os.listdir(Test_Cb_org)
    Test_Cr_org_list = os.listdir(Test_Cb_org)

    Test_Luma_path = Test_Luma_list
    Test_Chroma_path = (Test_Cb_Rec_list, Test_Cr_Rec_list, Test_QP_list)
    Test_org_path = (Test_Cb_org_list, Test_Cr_org_list)

    test = LumaAndChromaDataset(luma_path=Test_Luma_path, chroma_path=Test_Chroma_path,
                                org_path=Test_org_path, Train=False)

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    Luma_id, Chorma_id, org_id = next(iter(test_loader))
    print(Luma_id.shape, Chorma_id.shape, org_id.shape)
    #print(Luma_id[0, 3, :, :])
    

    Train_Luma_list = os.listdir(Train_Luma)
    Train_Cb_Rec_list = os.listdir(Train_Cb_rec)
    Train_Cr_Rec_list = os.listdir(Train_Cr_rec)
    Train_QP_list = os.listdir(Train_QP)

    Train_Cb_org_list = os.listdir(Train_Cb_org)
    Train_Cr_org_list = os.listdir(Train_Cr_org)

    Train_Luma_path = Train_Luma_list
    Train_Chroma_path = (Train_Cb_Rec_list, Train_Cr_Rec_list)
    Train_QP_path = Train_QP_list
    Train_org_path = (Train_Cb_org_list, Train_Cr_org_list)

    print('here1')

    test = LumaAndChromaDataset(luma_path=Train_Luma_path, chroma_path=Train_Chroma_path, qp_path=Train_QP_path,
                                org_path=Train_org_path, Train=True)

    print('here2')

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)

    Luma_id, Chorma_id, QP_id, org_id = next(iter(test_loader))
    print(Luma_id.shape, Chorma_id.shape, QP_id.shape, org_id.shape)
    '''

    description = {'Luma_id': 'byte', 'Chorma_id': 'byte', 'QP_id': 'byte', 'org_id': 'byte'}

    train = TFRecordDataset(Train_tfrecord,
                            index_path=None,
                            description=description,
                            transform=decode_image,
                            shuffle_queue_size=16384)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=4)

    data = next(iter(train_loader))
    print(data)
