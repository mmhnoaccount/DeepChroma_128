import numpy as np
import cv2
import os
import re
import torch.nn as nn
import torch


def tryint(s):  # 将元素中的数字转换为int后再排序
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):  # 将元素中的字符串和数字分割开
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def sort_humanly(v_list):  # 以分割后的list为单位进行排序
    return sorted(v_list, key=str2int)


def build_fullSize_Pic_luma(PicPackage):
    height = 1080 // 2
    width = 1920 // 2

    # A = np.zeros((height, width), dtype=float)
    A = np.zeros((1080, 1920), dtype=float)

    image_paths = os.listdir(PicPackage)
    image_paths = sort_humanly(image_paths)
    # print(image_paths)
    for j in range(width // 32):
        for i in range(height // 32):
            f = open(PicPackage + '/' + image_paths[j * (height // 32) + i])
            lines = f.readlines()
            '''
            block_row = i * 32
            for line in lines:
                per_line = line.strip('\n').split(' ')
                A[block_row, j * 32: j * 32 + 32] = per_line[:-1]
                block_row += 1
            '''
            block_row = i * 64
            count_line = 0
            for line in lines:
                count_line += 1

                # 当前块、左
                if count_line < 65:
                    continue

                '''
                # 上、左上
                if count_line > 64:
                    break
                '''

                per_line = line.strip('\n').split(' ')
                A[block_row, j * 64: j * 64 + 64] = per_line[64:-1]  # 当前块、上
                # A[block_row, j * 64: j * 64 + 64] = per_line[0:64]  #左、左上
                block_row += 1

    A = A / ((1 << 10) - 1)
    A = A * 255
    A = A.astype(np.uint8)
    return A


def visualize_results_luma(path, fix=True):
    Luma = build_fullSize_Pic_luma(path)
    return Luma


def build_fullSize_Pic_chroma(PicPackage):
    height = 1080 // 2
    width = 1920 // 2

    A = np.zeros((height, width), dtype=float)

    image_paths = os.listdir(PicPackage)
    image_paths = sort_humanly(image_paths)
    # print(image_paths)
    for j in range(width // 32):
        for i in range(height // 32):
            f = open(PicPackage + '/' + image_paths[j * (height // 32) + i])
            lines = f.readlines()
            block_row = i * 32
            for line in lines:
                per_line = line.strip('\n').split(' ')
                A[block_row, j * 32: j * 32 + 32] = per_line[:-1]
                block_row += 1

    A = A / ((1 << 10) - 1)
    A = A * 255
    A = A.astype(np.uint8)
    return A


def visualize_results_chroma(path, fix=True):
    Cb = build_fullSize_Pic_chroma(path)
    return Cb


def build_fullSize_Pic_LM(PicPackage):
    height = 1080 // 2
    width = 1920 // 2

    A = np.zeros((height, width), dtype=float)

    image_paths = os.listdir(PicPackage)
    image_paths = sort_humanly(image_paths)
    # print(image_paths)
    for j in range(width // 32):
        for i in range(height // 32):
            f = open(PicPackage + '/' + image_paths[j * (height // 32) + i])
            lines = f.readlines()
            '''
            block_row = i * 32
            for line in lines:
                per_line = line.strip('\n').split(' ')
                A[block_row, j * 32: j * 32 + 32] = per_line[:-1]
                block_row += 1
            '''
            block_row = i * 32
            count_line = 0
            for line in lines:
                count_line += 1

                # 当前块、左
                if count_line < 33:
                    continue

                '''
                # 上、左上
                if count_line > 32:
                    break
                '''

                per_line = line.strip('\n').split(' ')
                A[block_row, j * 32: j * 32 + 32] = per_line[32:-1]  # 当前块、上
                # A[block_row, j * 32: j * 32 + 32] = per_line[0:32]  #左、左上
                block_row += 1

    A = A / ((1 << 10) - 1)
    A = A * 255
    A = A.astype(np.uint8)
    return A


def visualize_results_LM(path, fix=True):
    Cr = build_fullSize_Pic_LM(path)
    return Cr

def build_fullSize_Pic(out_tensor, height=1080, width=1920, Luma=False):
    if not Luma:
        height = height // 2
        width = width // 2

    A = np.zeros((height, width), dtype=float)
    chan = 0
    if Luma:
        for j in range(width // 64):
            for i in range(height // 64):
                A[i * 64: i * 64 + 64, j * 64: j * 64 + 64] = out_tensor[chan, 64:128, 64:128]
                chan += 1
    else:
        for j in range(width // 32):
            for i in range(height // 32):
                A[i * 32: i * 32 + 32, j * 32: j * 32 + 32] = out_tensor[chan, :, :]
                chan += 1

    A = (A + 1) * 127.5
    A = A.astype(np.uint8)
    return A


'''
输入Y，U，V三张图
保存为IYUV420格式的图片
'''
def Save_IYUV420(Y, U, V):
    height, width = Y.shape

    U = U.reshape(-1, width)
    V = V.reshape(-1, width)
    ret_img = np.concatenate((Y, U, V), axis=0)

    return ret_img


if __name__ == '__main__':
    from network import Generator
    from loader.dataloader import visualize_Dataset
    from torch.utils.data import DataLoader

    compare_results = './compare_results/'


    net_opt = {
        'guide': True,
        # 'guide': False,     #noguide
        'relu': False,
        # 'relu': True,
        # 'bn': False,        #nobn
        'bn': True,
    }
    layers = [6, 4, 3, 3]
    # layers = [12, 8, 5, 5]
    G = Generator(luma_size=128, chroma_size=32,
                  luma_dim=1, output_dim=2, layers=layers, net_opt=net_opt)
    G = nn.DataParallel(G)
    checkpoint = torch.load(
        str(r'E:\File\package_by_mmh\DeepChroma\results\210513-220906\tag2pix_60_epoch.pkl'))  # 12000_per20
    G.load_state_dict(checkpoint['G'])

    G.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    visual = visualize_Dataset()
    visual_loader = DataLoader(visual, batch_size=visual.__len__(), shuffle=False)
    luma_in, chroma_in, QP_in, org_label = next(iter(visual_loader))
    print(luma_in.shape, chroma_in.shape, QP_in.shape)

    with torch.no_grad():

        if torch.cuda.is_available():
            luma_in, chroma_in, QP_in = luma_in.to(device), chroma_in.to(device), QP_in.to(device)

        G_f = G(luma_in, chroma_in, QP_in)

        if torch.cuda.is_available():
            G_f = G_f.cpu()
            luma_in = luma_in.cpu()

        # print(G_f.shape)
        # print(org_label.shape)

        height = 1080
        width = 1920

        luma = build_fullSize_Pic(luma_in[:, 0, :, :].view(visual.__len__(), 128, 128), height, width, Luma=True)
        pre_Cb = build_fullSize_Pic(G_f[:, 0, :, :].view(visual.__len__(), 32, 32), height, width)
        pre_Cr = build_fullSize_Pic(G_f[:, 1, :, :].view(visual.__len__(), 32, 32), height, width)
        # pre_Cb = build_fullSize_Pic(org_label[:, 0, :, :].view(visual.__len__(), 32, 32), height, width)
        # pre_Cr = build_fullSize_Pic(org_label[:, 1, :, :].view(visual.__len__(), 32, 32), height, width)

        pre_img = Save_IYUV420(luma, pre_Cb, pre_Cr)

        path = compare_results + 'DeepPred.yuv'
        f = open(path, "wb")
        pre_img.tofile(f)
        f.close()


        # Y = visualize_results_luma(r'E:\File\package_by_mmh\train_set_128\visualize_valid\Luma')
        U_org = visualize_results_chroma(r'E:\File\package_by_mmh\train_set_128\visualize_valid\testVisualize_set\origCb')
        V_org = visualize_results_chroma(r'E:\File\package_by_mmh\train_set_128\visualize_valid\testVisualize_set\origCr')

        YUV_org = Save_IYUV420(luma, U_org, V_org)
        path = compare_results + 'orig.yuv'
        f = open(path, "wb")
        YUV_org.tofile(f)
        f.close()

        U_LM = visualize_results_LM(r'E:\File\package_by_mmh\train_set_128\visualize_valid\testVisualize_set\RecCb')
        V_LM = visualize_results_LM(r'E:\File\package_by_mmh\train_set_128\visualize_valid\testVisualize_set\RecCr')
        YUV_LM = Save_IYUV420(luma, U_LM, V_LM)
        path = compare_results + 'LM.yuv'
        f = open(path, "wb")
        YUV_LM.tofile(f)
        f.close()
