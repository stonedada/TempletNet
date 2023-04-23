import argparse
import logging
import sys
from datetime import datetime
from torch.autograd import Variable
from torch import optim, nn
from torch.nn import MSELoss, L1Loss
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
from inference import *
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils.config import get_device

from models.TransformerUNetParallel import TransformerUNetParallel
from models.TransformerUNet import TransformerUNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/stone/dataset/npy_1024/train',
                    help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='/home/dataset/npy_128/test',
                    help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--label_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=4, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=128, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true",
                    help='whether to save results during inference')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
args = parser.parse_args()


def train(train_loader, model, optimizer, epoch, best_loss, snapshot_path):
    global iter_num
    loss_sum = 0
    loss_total = 0
    for i, sampled_batch in enumerate(train_loader, start=1):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        # image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        image_batch, label_batch = Variable(image_batch.to(device=device)), label_batch.to(device='cuda:0')
        bt_size = image_batch.size(0)
        # ---- forward ----
        outputs = model(image_batch)
        # ---- loss function ----
        loss = mae_loss(outputs, label_batch)
        # ---- backward ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        # record loss
        loss = loss.detach()
        l = loss.item()
        loss_sum += l * bt_size
        loss_total += bt_size

        iter_num = iter_num + 1
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/total_loss', loss, iter_num)

        logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[loss: {:.4f}]'.
                  format(datetime.now(), epoch, args.max_epochs, i, total_step,
                         loss.data))

    if (epoch + 1) % 1 == 0:
        meanloss = test(args, model)
        if meanloss < best_loss:
            print('new best loss: ', meanloss)
            best_loss = meanloss
            torch.save(model.state_dict(), snapshot_path + f'/{args.model_name}-%d.pth' % epoch)
            print('[Saving Snapshot:]', snapshot_path + f'/{args.model_name}-%d.pth' % epoch)
    return best_loss, loss_sum / loss_total


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '/home/dataset/npy_256/train',
            'label_dir': '/home/dataset/npy_256/train_label',
            'num_classes': 9,
        },
    }

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.label_dir = dataset_config[dataset_name]['label_dir']

    # ---- build models ----

    # change to your model #
    channels = (3, 32, 64, 128, 256, 512)
    is_residual = True
    bias = True
    heads = 4
    size = (128, 128)

    # channels = (3, 32, 64, 128)
    # is_residual = True
    # bias = True
    # heads = 4
    # size = (128, 128)

    model_path = f'res_{is_residual}_head_{heads}_ch_{channels[-1]}'
    device = get_device()
    print(device, torch.cuda.device_count())

    timestamp = time.time()
    tupletime = time.localtime(timestamp)
    day_time = str(tupletime[1]) + '_' + str(tupletime[2]) + '_' + str(tupletime[3])

    args.day_time = day_time
    args.model_path = model_path
    args.device = device
    args.size = size
    args.model_name = "UTransform"
    # args.model_name = "TransformUNet"
    args.exp = 'TU_' + dataset_name + str(args.size[0])
    snapshot_path = "../model/{}/{}".format(args.exp, args.model_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    model = TransformerUNetParallel(channels, heads, size[0], is_residual, bias)
    # model = TransformerUNet(channels, heads, is_residual, bias)
    from datasets.dataset_npy import Synapse_dataset, RandomGenerator

    logging.basicConfig(filename=snapshot_path + f"/{args.model_name}_train_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu
    print("batch_size", batch_size)

    db_train = Synapse_dataset(base_dir=args.root_path, label_dir=args.label_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=size)]))
    print("The length of train set is: {}".format(len(db_train)))


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    train_loader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=4, pin_memory=True,
                              worker_init_fn=worker_init_fn)

    # if args.n_gpu > 1:
    #     model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # model.to(device=torch.device('cuda', 0))

    mse_loss = MSELoss()
    mae_loss = L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/Tensorboard_log')
    total_step = len(train_loader)
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    best_loss = 1e5
    iter_num = 0
    train_loss_history = []
    model.train()
    for epoch in range(max_epoch):
        best_loss, t = train(train_loader, model, optimizer, epoch, best_loss, snapshot_path)
        train_loss_history.append(t)
    writer.close()
    np.save('./UTransform_train_loss.npy', np.array(train_loss_history))