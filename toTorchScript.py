import torch
import torchvision
from network import Generator
import torch.nn as nn


# An instance of your model.
#model = lstm_model()
#model.load_state_dict(torch.load('ccc_model_50.pkl'))
net_opt = {
            'guide': True,
            #'guide': False,     #noguide
            'relu': False,
            #'relu': True,
            #'bn': False,        #nobn
            'bn': True,
        }
layers = [6, 4, 3, 3]
#layers = [12, 8, 5, 5]
G = Generator(luma_size=128, chroma_size=32,
                           luma_dim=1, output_dim=2, layers=layers, net_opt=net_opt)
G = nn.DataParallel(G)
checkpoint = torch.load(str(r'E:\File\package_by_mmh\results\210514-154618_lr0.0002_9000_per10\tag2pix_60_epoch.pkl'))
G.load_state_dict(checkpoint['G'])
G = G.module
device = torch.device("cpu")
G = G.to(device)



# An example input you would normally provide to your model's forward() method.
#example_in = torch.ones([1, 1, 1])
#example_h0 = (torch.zeros([1, 1, 1]), torch.zeros([1, 1, 1]))
#print(example_h0.shape)
example_Luma = torch.ones([1, 1, 128, 128])
example_Chroma = torch.zeros([1, 2, 64, 64])
example_QP = torch.zeros([1, 1, 32, 32])


# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
#traced_script_module = torch.jit.trace(model, (example_in, example_h0))
#traced_script_module.save("Script_ccc_model_50.pt")
traced_script_module = torch.jit.trace(G, (example_Luma, example_Chroma, example_QP))
traced_script_module.save(r"E:\File\package_by_mmh\results\210514-154618_lr0.0002_9000_per10\tag2pix_60_epoch.pt")