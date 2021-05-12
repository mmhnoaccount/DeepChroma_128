import itertools, time, pickle, pprint
from pathlib import Path
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import utils
from loader.dataloader import get_dataset, visualize_Dataset
from torch.utils.data import DataLoader
from network import Discriminator



class tag2pix(object):
    def __init__(self, args):
        if args.model == 'tag2pix':
            from network import Generator
        elif args.model == 'senet':
            from model.GD_senet import Generator
        elif args.model == 'resnext':
            from model.GD_resnext import Generator
        elif args.model == 'catconv':
            from model.GD_cat_conv import Generator
        elif args.model == 'catall':
            from model.GD_cat_all import Generator
        elif args.model == 'adain':
            from model.GD_adain import Generator
        elif args.model == 'seadain':
            from model.GD_seadain import Generator
        else:
            raise Exception('invalid model name: {}'.format(args.model))

        self.args = args
        self.epoch = args.epoch
        self.batch_size = args.batch_size

        self.gpu_mode = not args.cpu
        self.input_size = args.input_size
        self.layers = args.layers

        self.load_dump = (args.load is not "")

        self.load_path = Path(args.load)

        self.l2_lambda = args.l2_lambda
        self.guide_beta = args.guide_beta
        #self.adv_lambda = args.adv_lambda
        self.save_freq = args.save_freq

        self.use_visualize = args.use_visualize
        self.use_tfrecord = args.use_tfrecord

        #self.two_step_epoch = args.two_step_epoch
        #self.brightness_epoch = args.brightness_epoch
        self.save_all_epoch = args.save_all_epoch

        self.start_epoch = 1

        #### load dataset
        if not args.test:
            # self.train_data_loader, self.test_data_loader = get_dataset(args)
            self.train_data_loader = get_dataset(args)
            self.result_path = Path(args.result_dir) / time.strftime('%y%m%d-%H%M%S', time.localtime())

            if not self.result_path.exists():
                self.result_path.mkdir()

            #self.test_images = self.get_test_data(self.test_data_loader, args.test_image_count)
        else:
            self.test_data_loader = get_dataset(args)
            self.result_path = Path(args.result_dir)


        ##### initialize network
        self.net_opt = {
            'guide': not args.no_guide,
            'relu': args.use_relu,
            'bn': not args.no_bn,
            #'cit': not args.no_cit
        }

        self.G = Generator(luma_size=args.input_size, chroma_size=32,
                           luma_dim=1, output_dim=2, layers=args.layers, net_opt=self.net_opt)
        self.D = Discriminator(input_dim=2, output_dim=1, input_size=self.input_size)

        if args.test:
            for param in self.G.parameters():
                param.requires_grad = False
            for param in self.D.parameters():
                param.requires_grad = False

        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        self.BCE_loss = nn.BCELoss()
        self.MSE_Loss = nn.MSELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("gpu mode: ", self.gpu_mode)
        print("device: ", self.device)
        print(torch.cuda.device_count(), "GPUS!")

        if self.gpu_mode:
            self.G.to(self.device)
            self.D.to(self.device)

            self.BCE_loss.to(self.device)
            self.MSE_Loss.to(self.device)

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.to(self.device), self.y_fake_.to(self.device)

        if self.load_dump:
            self.load(self.load_path)

            print("continue training!!!!")
        else:
            self.end_epoch = self.epoch

        self.print_params()

        self.D.train()
        print('training start!!')
        start_time = time.time()

        for epoch in range(self.start_epoch, self.end_epoch + 1):
            print("EPOCH: {}".format(epoch))

            self.G.train()
            epoch_start_time = time.time()

            if self.use_tfrecord:
                max_iter = 1363086 // self.batch_size
            else:
                max_iter = self.train_data_loader.dataset.__len__() // self.batch_size

            # for iter, (luma_in, chroma_in, qp_in, org_label) in enumerate(tqdm(self.train_data_loader, ncols=80)):
            for iter, data in enumerate(tqdm(self.train_data_loader, ncols=80, total=max_iter)):
                if iter >= max_iter:
                    break

                if self.use_tfrecord:
                    luma_in, chroma_in, qp_in, org_label = data.values()

                if self.gpu_mode:
                    luma_in, chroma_in, qp_in, org_label = luma_in.to(self.device), chroma_in.to(self.device), qp_in.to(self.device), org_label.to(self.device)

                # update D network
                if iter % 5 == 0:
                    self.D_optimizer.zero_grad()

                    D_real = self.D(org_label)
                    D_real_loss = self.BCE_loss(D_real, self.y_real_)

                    G_f = self.G(luma_in, chroma_in, qp_in)
                    if self.gpu_mode:
                        G_f = G_f.to(self.device)

                    D_f_fake = self.D(G_f)
                    D_f_fake_loss = self.BCE_loss(D_f_fake, self.y_fake_)

                    D_loss = D_real_loss + D_f_fake_loss

                    self.train_hist['D_loss'].append(D_loss.item())

                    D_loss.backward()
                    self.D_optimizer.step()
                else:
                    self.train_hist['D_loss'].append(self.train_hist['D_loss'][-1])

                # update G network
                self.G_optimizer.zero_grad()

                G_f = self.G(luma_in, chroma_in, qp_in)

                if self.gpu_mode:
                    G_f = G_f.to(self.device)

                D_f_fake = self.D(G_f)

                D_f_fake_loss = self.BCE_loss(D_f_fake, self.y_real_)

                L2_D_f_fake_loss = self.MSE_Loss(G_f, org_label)
                # L2_D_g_fake_loss = self.MSE_Loss(G_g, org_label) if self.net_opt['guide'] else 0

                # G_loss = D_f_fake_loss + (L2_D_f_fake_loss + L2_D_g_fake_loss * self.guide_beta) * self.l2_lambda
                G_loss = D_f_fake_loss + L2_D_f_fake_loss * self.l2_lambda

                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [{:2d}] [{:4d}/{:4d}] D_loss: {:.8f}, G_loss: {:.8f}".format(
                        epoch, (iter + 1), max_iter, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            with torch.no_grad():
                #self.visualize_results(epoch)
                if self.use_visualize:
                    self.visualize_results(epoch)
                utils.loss_plot(self.train_hist, self.result_path, epoch)

            if epoch >= self.save_all_epoch > 0:
                self.save(epoch)
            elif self.save_freq > 0 and epoch % self.save_freq == 0:
                self.save(epoch)

        print("Training finish!... save training results")

        if self.save_freq == 0 or epoch % self.save_freq != 0:
            if self.save_all_epoch <= 0 or epoch < self.save_all_epoch:
                self.save(epoch)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: {:.2f}, total {} epochs time: {:.2f}".format(
            np.mean(self.train_hist['per_epoch_time']), self.epoch, self.train_hist['total_time'][0]))


    def test(self):

        self.load_test(self.args.load)

        self.D.eval()
        self.G.eval()

        load_path = self.load_path
        result_path = self.result_path / load_path.stem

        if not result_path.exists():
            result_path.mkdir()

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.to(self.device), self.y_fake_.to(self.device)

        with torch.no_grad():
            for iter, (luma_in, chroma_in, qp_in, org_label) in enumerate(tqdm(self.test_data_loader, ncols=80)):
                if self.gpu_mode:
                    luma_in, chroma_in, qp_in, org_label = luma_in.to(self.device), chroma_in.to(self.device), qp_in.to(self.device), org_label.to(self.device)

                D_real = self.D(org_label)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_f = self.G(luma_in, chroma_in, qp_in)
                if self.gpu_mode:
                    G_f = G_f.to(self.device)

                D_f_fake = self.D(G_f)
                D_f_fake_loss = self.BCE_loss(D_f_fake, self.y_fake_)

                D_loss = D_real_loss + D_f_fake_loss


                G_f = self.G(luma_in, chroma_in, qp_in)

                if self.gpu_mode:
                    G_f= G_f.to(self.device)

                D_f_fake = self.D(G_f)

                D_f_fake_loss = self.BCE_loss(D_f_fake, self.y_real_)

                L2_D_f_fake_loss = self.MSE_Loss(G_f, org_label)
                # L2_D_g_fake_loss = self.MSE_Loss(G_g, org_label) if self.net_opt['guide'] else 0

                #G_loss = D_f_fake_loss + (L2_D_f_fake_loss + L2_D_g_fake_loss * self.guide_beta) * self.l2_lambda
                G_loss = D_f_fake_loss + L2_D_f_fake_loss * self.l2_lambda


                if ((iter + 1) % 100) == 0:
                    print("[{:4d} D_loss: {:.8f}, G_loss: {:.8f}".format(
                        (iter + 1), D_loss.item(), G_loss.item()))

        #self.visualize_results(0)



    def visualize_results(self, epoch, fix=True):
        if not self.result_path.exists():
            self.result_path.mkdir()

        self.G.eval()

        visual = visualize_Dataset()
        visual_loader = DataLoader(visual, batch_size=visual.__len__(), shuffle=False)
        luma_in, chroma_in, QP_in, org_label = next(iter(visual_loader))

        with torch.no_grad():
            
            if self.gpu_mode:
                luma_in, chroma_in, QP_in = luma_in.to(self.device), chroma_in.to(self.device), QP_in.to(self.device)

            G_f = self.G(luma_in, chroma_in)

            if self.gpu_mode:
                G_f = G_f.cpu()
                luma_in = luma_in.cpu()

            #print(G_f.shape)
            #print(org_label.shape)

            height = 1352
            width = 1352

            luma = self.build_fullSize_Pic(luma_in[:, 0, :, :].view(visual.__len__(), 128, 128), height, width, Luma=True)
            pre_Cb = self.build_fullSize_Pic(G_f[:, 0, :, :].view(visual.__len__(), 32, 32), height, width)
            pre_Cr = self.build_fullSize_Pic(G_f[:, 1, :, :].view(visual.__len__(), 32, 32), height, width)

            pre_Cb = pre_Cb.reshape(-1, width)
            pre_Cr = pre_Cr.reshape(-1, width)
            pre_img = np.concatenate((luma, pre_Cb, pre_Cr), axis=0)

            path = str(self.result_path) + '/' + 'tag2pix_epoch' + str(epoch) + '.yuv'
            f = open(path, "wb")
            pre_img.tofile(f)
            f.close()

            '''
            org_Cb = self.build_fullSize_Pic(org_label[:, 0, :, :].view(visual.__len__(), 32, 32), height, width)
            org_Cr = self.build_fullSize_Pic(org_label[:, 1, :, :].view(visual.__len__(), 32, 32), height, width)

            
            YUVpre = np.concatenate((np.expand_dims(luma, 0), np.expand_dims(pre_Cb, 0), np.expand_dims(pre_Cr, 0)), 0)
            YUVpre = YUVpre.transpose((1, 2, 0))
            img_pre = cv2.cvtColor(YUVpre, cv2.COLOR_YUV2BGR)
            path = str(self.result_path) + '/' + 'tag2pix_epoch' + str(epoch) + '.png'
            cv2.imwrite(path, img_pre)
            '''

            '''
            cv2.imshow('img', img_pre)
            cv2.waitKey(0)
            #cv2.imwrite(self.result_path / 'tag2pix_epoch{:03d}_G_f.png'.format(epoch), img_pre)

            YUVorg = np.concatenate((np.expand_dims(luma, 0), np.expand_dims(org_Cb, 0), np.expand_dims(org_Cr, 0)), 0)
            YUVorg = YUVorg.transpose((1, 2, 0))
            img_org = cv2.cvtColor(YUVorg, cv2.COLOR_YUV2BGR)
            cv2.imshow('img', img_org)
            cv2.waitKey(0)
            #cv2.imwrite(self.result_path / 'tag2pix_epoch{:03d}_G_f.png'.format(epoch), img_org)
            path = './results/' + 'tag2pix_epoch' + str(epoch) + '.png'
            cv2.imwrite(path, img_org)

            Cb_CCLM = self.build_fullSize_Pic(luma_in[:, 1, :, :].view(visual.__len__(), 32, 32))
            Cr_CCLM = self.build_fullSize_Pic(luma_in[:, 2, :, :].view(visual.__len__(), 32, 32))
            YUVCCLM = np.concatenate((np.expand_dims(luma, 0), np.expand_dims(Cb_CCLM, 0), np.expand_dims(Cr_CCLM, 0)), 0)
            YUVCCLM = YUVCCLM.transpose((1, 2, 0))
            img_CCLM = cv2.cvtColor(YUVCCLM, cv2.COLOR_YUV2BGR)
            cv2.imshow('img', img_CCLM)
            cv2.waitKey(0)
            '''

    def build_fullSize_Pic(self, out_tensor, height=1080, width=1920, Luma=False):
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


    def save(self, save_epoch):
        if not self.result_path.exists():
            self.result_path.mkdir()

        with (self.result_path / 'arguments.txt').open('w') as f:
            f.write(pprint.pformat(self.args.__dict__))
        
        save_dir = self.result_path

        torch.save({
            'G' : self.G.state_dict(),
            'D' : self.D.state_dict(),
            'G_optimizer' : self.G_optimizer.state_dict(),
            'D_optimizer' : self.D_optimizer.state_dict(),
            'finish_epoch' : save_epoch,
            'result_path' : str(save_dir)
            }, str(save_dir / 'tag2pix_{}_epoch.pkl'.format(save_epoch)))

        with (save_dir / 'tag2pix_{}_history.pkl'.format(save_epoch)).open('wb') as f:
            pickle.dump(self.train_hist, f)

        print("============= save success =============")
        print("epoch from {} to {}".format(self.start_epoch, save_epoch))
        print("save result path is {}".format(str(self.result_path)))

    def load_test(self, checkpoint_path):
        print(checkpoint_path)
        print(self.net_opt)
        checkpoint = torch.load(str(checkpoint_path))
        self.G.load_state_dict(checkpoint['G'])

        '''
        example_Luma = torch.ones([1, 4, 32, 32])
        example_Chroma = torch.zeros([1, 2, 32, 32])
        example_Ref = torch.zeros([1, 128])
        traced_script_module = torch.jit.trace(self.G, (example_Luma, example_Chroma, example_Ref))
        traced_script_module.save("Script_model_no_Bn_epoch50.pt")
        '''

    def load(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path))
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
        self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        self.start_epoch = checkpoint['finish_epoch'] + 1

        self.finish_epoch = self.args.epoch + self.start_epoch - 1
        self.end_epoch = self.finish_epoch

        print("============= load success =============")
        print("epoch start from {} to {}".format(self.start_epoch, self.finish_epoch))
        print("previous result path is {}".format(checkpoint['result_path']))

    '''
    def get_test_data(self, test_data_loader, count):
        test_count = 0
        original_, sketch_, iv_tag_, cv_tag_ = [], [], [], []
        for orig, sket, ivt, cvt in test_data_loader:
            original_.append(orig)
            sketch_.append(sket)
            iv_tag_.append(ivt)
            cv_tag_.append(cvt)

            test_count += len(orig)
            if test_count >= count:
                break

        original_ = torch.cat(original_, 0)
        sketch_ = torch.cat(sketch_, 0)
        iv_tag_ = torch.cat(iv_tag_, 0)
        cv_tag_ = torch.cat(cv_tag_, 0)
        
        self.save_tag_tensor_name(iv_tag_, cv_tag_, self.result_path / "test_image_tags.txt")

        image_frame_dim = int(np.ceil(np.sqrt(len(original_))))

        if self.gpu_mode:
            original_ = original_.cpu()
        sketch_np = sketch_.data.numpy().transpose(0, 2, 3, 1)
        original_np = self.color_revert(original_)

        utils.save_images(original_np[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        self.result_path / 'tag2pix_original.png')
        utils.save_images(sketch_np[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        self.result_path / 'tag2pix_sketch.png')

        return original_, sketch_, iv_tag_, cv_tag_
    

    def save_tag_tensor_name(self, iv_tensor, cv_tensor, save_file_path):
        #iv_tensor, cv_tensor: batched one-hot tag tensors    
        iv_dict_inverse = {tag_index: tag_id for (tag_id, tag_index) in self.iv_dict.items()}
        cv_dict_inverse = {tag_index: tag_id for (tag_id, tag_index) in self.cv_dict.items()}

        with open(save_file_path, 'w') as f:
            f.write("CIT tags\n")

            for tensor_i, batch_unit in enumerate(iv_tensor):
                tag_list = []
                f.write(f'{tensor_i} : ')

                for i, is_tag in enumerate(batch_unit):
                    if is_tag:
                        tag_name = self.id_to_name[iv_dict_inverse[i]]
                        tag_list.append(tag_name)
                        f.write(f"{tag_name}, ")
                f.write("\n")

            f.write("\nCVT tags\n")

            for tensor_i, batch_unit in enumerate(cv_tensor):
                tag_list = []
                f.write(f'{tensor_i} : ')

                for i, is_tag in enumerate(batch_unit):
                    if is_tag:
                        tag_name = self.id_to_name[cv_dict_inverse[i]]
                        tag_list.append(self.id_to_name[cv_dict_inverse[i]])
                        f.write(f"{tag_name}, ")
                f.write("\n")
    '''
    def print_params(self):
        params_cnt = [0, 0, 0]
        for param in self.G.parameters():
            params_cnt[0] += param.numel()
        for param in self.D.parameters():
            params_cnt[1] += param.numel()
        '''
        for param in self.Pretrain_ResNeXT.parameters():
            params_cnt[2] += param.numel()
        '''
        print(f'Parameter #: G - {params_cnt[0]} / D - {params_cnt[1]}')
