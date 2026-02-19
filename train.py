from utils.tools import *
# from model.network import *
from torch.autograd import Variable
# import os
import torch
import torch.optim as optim
import time
# import numpy as np
from loguru import logger
from model.nest import Nest
from model.vit import VisionTransformer, CONFIGS
# from torch.autograd import Variable
from ptflops import get_model_complexity_info
from apex import amp
torch.multiprocessing.set_sharing_strategy('file_system')
from model.label_net import LabelNet
from utils.loss import *
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
from MSLOSS import MultiSimilarityLoss


def get_config():
    config = {
        "alpha": 0.1,
         "beta": 0.2,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 1e-5}, "lr_type": "step"},
        "info": "[HTMSH]",
        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
         # "dataset": "UCMD",
        "dataset": "AID",
        #  "dataset": "WHURS19",
        "Label_dim": 30, # nclass
        "thresh":  0.5,
        "margin": 0.5,
        "α": 1,
        "epoch": 200,
        "test_map": 0,
        "save_path": "save/HashNet",
        "device": torch.device("cuda:0"),
        'test_device': torch.device("cuda:0"),
        "bit_list": [48],
        "pretrained_dir": "jx_nest_base-8bc41011.pth",
        "img_size": 224,
        "patch_size": 4,
        "in_chans": 3,
        "num_work": 10,
        "model_type": "ViT-L_16",
        "top_img": 10
    }
    config = config_dataset(config)
    return config


def train_val(config, bit):
    configs = CONFIGS[config["model_type"]]
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["bit"] = bit

    net = Nest(config, num_levels=3, embed_dims=(128, 256, 512), num_heads=(4, 8, 16), depths=(2, 2, 14))
    L_net = LabelNet(code_len=bit, label_dim=config["Label_dim"])
    if config["pretrained_dir"] is not None:
        logger.info('Loading:', config["pretrained_dir"])
        state_dict = torch.load(config["pretrained_dir"])
        net.load_state_dict(state_dict, strict=False)
        logger.info('Pretrain weights loaded.')

    net.to(config["device"])
    L_net.to(config["device"])

    flops, num_params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    logger.info("Total Parameter: \t%s" % num_params)
    logger.info("Total Flops: \t%s" % flops)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    L_optimizer = torch.optim.Adam(L_net.parameters(), lr=1e-5)

    msloss = MultiSimilarityLoss(thresh = config["thresh"],margin = config["margin"],scale_pos = 2.0,scale_neg=40.0)

    Best_mAP = 0
    [net, L_net], [optimizer, L_optimizer] = amp.initialize(models=[net, L_net],
                                                                            optimizers=[optimizer, L_optimizer],
                                                                            opt_level='O2',
                                                                            num_losses=2)
    amp._amp_state.loss_scalers[0]._loss_scale = 5
    amp._amp_state.loss_scalers[1]._loss_scale = 1

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        logger.info("%s[%2d/%2d][%s] bit:%d, datasets:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        L_net.train()
        L_net.set_alpha(epoch)
        logger.info('Epoch [%d/%d], alpha for LabelNet: %.3f' % (epoch + 1, config["epoch"], L_net.alpha))
        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            L_optimizer.zero_grad()

            _, _, label_output = L_net(label.to(torch.float32))

            ss_ = (label.to(torch.float64) @ label.to(torch.float64).t() > 0) * 2 - 1

            u = net(image)
            u = F.normalize(u)

            label = label.float()
            ms_loss = msloss(u, label)

            loss = torch.mean(torch.square((torch.matmul(u, label_output.t())) - ss_))

            Total_Loss = ms_loss + config["α"]*loss

            train_loss += loss.item() + ms_loss.item()

            with amp.scale_loss(Total_Loss, optimizer, loss_id=0) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
            with amp.scale_loss(loss,  L_optimizer, loss_id=1) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()
            L_optimizer.step()
            mslossitem = ms_loss.item()
            lossitem = loss.item()

        train_loss = train_loss / len(train_loader)
        mslossitem = mslossitem / len(train_loader)
        lossitem = lossitem / len(train_loader)

        logger.info("\b\b\b\b\b\b\b train_loss:%.4f" % (train_loss))
        logger.info("\b\b\b\b\b\b\b msloss:%.4f" % (mslossitem))
        logger.info("\b\b\b\b\b\b\b loss:%.4f" % (lossitem))

        if (epoch + 1) > config["test_map"]:
            Best_mAP, index_img = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, 10)
            net.to(config["device"])

if __name__ == "__main__":
    config = get_config()

    logger.add('logs/{time}' + config["info"] + '_' + config["dataset"] + '_' +' α '+str(config["α"]) + ' thresh ' + str(config["thresh"])+ ' Margin ' + str(config["margin"]) + ' alpha ' +str(config["alpha"])+ '.log',
               rotation='50 MB', level='DEBUG')

    logger.info(config)
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/HTMSH_{config['dataset']}_{bit}.json"
        train_val(config, bit)