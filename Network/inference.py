import argparse
import logging
import os
import sys
import time
import tifffile
import torch
from torch.nn.modules.loss import MSELoss, L1Loss
from torchvision import transforms
from torch.autograd import Variable
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from skimage import metrics
from datasets.dataset_npy import Synapse_dataset, RandomGenerator
from utils.util import make_dataframe, DF_NAMES
from utils.custom_metrics import *
from utils.config import get_device
from models.TransformerUNetParallel import TransformerUNetParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/home/stone/dataset/npy_128/test',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--n_gpu', type=int, default=5, help='total gpu')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=128, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_false", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def test(args, model):
    model.eval()
    mean_loss = []
    mae_loss = L1Loss()
    for s in ['val', 'test']:
        image_root = '{}/{}'.format(args.volume_path, s)
        gt_root = '{}/{}_label'.format(args.volume_path, s)
        save_path = test_save_path if args.is_savenii else os.path.join('../predictions', args.model_path,
                                                                        f"{s}_{args.day_time}")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        db_train = Synapse_dataset(base_dir=image_root, label_dir=gt_root, split="train",
                                   transform=transforms.Compose(
                                       [RandomGenerator(output_size=args.size)]))
        print("The length of test set is: {}".format(len(db_train)))

        def worker_init_fn(worker_id):
            random.seed(args.seed + worker_id)

        trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=4, pin_memory=True,
                                 worker_init_fn=worker_init_fn)

        iter_num = 0

        loss_bank = []
        nrmse_bank = []
        ssim_bank = []
        dice_bank = []
        iou_bank = []
        pcc_bank = []
        r2_bank = []

        frames_meta = make_dataframe(nbr_rows=len(trainloader))
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(trainloader):
                image_batch, label_batch, case_name = sampled_batch['image'], sampled_batch['label'], \
                    sampled_batch['case_name'][0]
                image_batch, label_batch = Variable(image_batch.to(device=args.device)), label_batch.to(device='cuda:0')
                outputs = model(image_batch)

                x_true = torch.squeeze(label_batch).cpu().numpy()
                x_pre = torch.squeeze(outputs).cpu().numpy()

                # evaluate
                loss = mae_loss(outputs, label_batch)
                dice = dice_metric(x_true, x_pre)
                mIoU = iou_metric(x_true, x_pre)
                nrmse = metrics.normalized_root_mse(x_true, x_pre)
                ssim = metrics.structural_similarity(x_true, x_pre, win_size=21,
                                                     data_range=x_true.max() - x_true.min())
                pcc = pearsonr(x_pre.flatten(), x_true.flatten())
                r2 = r2_metric(x_true, x_pre)

                # Save tif file
                prediction = torch.squeeze(outputs, dim=0).cpu().numpy()
                _image = torch.squeeze(image_batch).cpu().numpy()
                _label = torch.squeeze(label_batch, dim=0).cpu().numpy()

                tifffile.imwrite(f'{save_path}/{case_name}_pred.tif', data=prediction)
                tifffile.imwrite(f'{save_path}/{case_name}_image.tif', data=_image)
                tifffile.imwrite(f'{save_path}/{case_name}_label.tif', data=_label)

                # Save csv.file
                meta_row = dict.fromkeys(DF_NAMES)
                meta_row['MAE'] = loss.item()
                meta_row['NRMSE'] = nrmse
                meta_row['SSIM'] = ssim
                meta_row['PCC'] = pcc
                meta_row['Dice'] = dice
                meta_row['mIOU'] = mIoU
                meta_row['r2'] = r2
                frames_meta.loc[iter_num] = meta_row
                iter_num = iter_num + 1

                # Record
                loss_bank.append(loss.item())
                nrmse_bank.append(nrmse)
                pcc_bank.append(pcc)
                dice_bank.append(dice)
                iou_bank.append(mIoU)
                ssim_bank.append(ssim)
                r2_bank.append(r2)

                logging.info('iteration %d : MAELoss: %.4f,NRMSE: %f,ssim: %f,PCC: %f,dice: %f,mIoU: %f,R2: %f' % (
                    iter_num, loss.item(), nrmse, ssim, pcc, dice, mIoU, r2))

            frames_meta_filename = os.path.join("../predictions", args.model_path,
                                                f"{s}_inference_{args.day_time}.csv")
            frames_meta.to_csv(frames_meta_filename, sep=',')

            print('{} MAELoss: {:.4f}, NRMSE: {:.4f}, Dice: {:.4f}, IoU: {:.4f}, ssim:{:.4f}, PCC:{:.4f}, R2:{:.4f}'.
                  format(s, np.mean(loss_bank), np.mean(nrmse_bank), np.mean(dice_bank), np.mean(iou_bank),
                         np.mean(ssim_bank), np.mean(pcc_bank), np.mean(r2_bank)))
            mean_loss.append(np.mean(loss_bank))

    return mean_loss[0]


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

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '/home/dataset/npy_256',
            'label_dir': '/home/dataset/npy_256',
            'num_classes': 9,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.label_dir = dataset_config[dataset_name]['label_dir']

    # ---- build new models ----

    channels = (3, 32, 64, 128, 256, 512)
    is_residual = True
    bias = True
    heads = 4
    size = (128, 128)
    task = 'F-actin'
    # task = 'nuclei'

    model_path = f'res_{is_residual}_head_{heads}_ch_{channels[-1]}_{task}'

    device = get_device()
    print(device, torch.cuda.device_count())

    args.model_path = model_path
    args.device = device
    args.size = size

    # name the same snapshot defined in train script!
    args.model_name = "UTransform"
    args.exp = 'TU_' + dataset_name + str(args.size[0])
    snapshot_path = "../model/{}/{}".format(args.exp, args.model_path)

    model = TransformerUNetParallel(channels, heads, size[0], is_residual, bias)
    # load checkpoint
    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', f'{args.model_name}-' + str(102))
    print('snapshot', snapshot, os.path.exists(snapshot))

    model.load_state_dict(torch.load(snapshot), strict=False)
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    timestamp = time.time()
    tupletime = time.localtime(timestamp)
    day_time = str(tupletime[1]) + '_' + str(tupletime[2]) + '_' + str(tupletime[3])
    args.day_time = day_time

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.model_path, f"test_{args.model_name}_{args.day_time}")
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    test(args, model)
