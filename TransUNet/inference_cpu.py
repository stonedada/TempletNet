import argparse
import logging
import os
import sys
import time
import tifffile
from torch.nn.modules.loss import MSELoss
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
from models.TransformerUNet import TransformerUNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/home/stone/dataset/npy_224/test',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--label_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
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
    save_path = test_save_path if args.is_savenii else os.path.join('../predictions', args.model_path,
                                                                    f"test_{args.model_name}_{args.day_time}")
    os.makedirs(save_path, exist_ok=True)

    db_train = Synapse_dataset(base_dir=args.volume_path, label_dir=args.label_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=args.size)]))
    print("The length of test set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    model.eval()
    mse_loss = MSELoss()
    iter_num = 0
    thresh = 1

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
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            # image_batch, label_batch = Variable(image_batch.to(device=args.device)), label_batch.to(device='cuda:0')
            outputs = model(image_batch)

            x_true = torch.squeeze(label_batch).cpu().numpy()
            x_pre = torch.squeeze(outputs).cpu().numpy()

            loss = mse_loss(outputs, label_batch)
            dice = mean_dice(x_true, x_pre, thresh)
            mIoU = mean_iou(x_true, x_pre, thresh)
            nrmse = metrics.normalized_root_mse(x_true, x_pre)
            mse = metrics.mean_squared_error(x_true, x_pre)
            ssim_val = metrics.structural_similarity(x_true, x_pre)
            pcc = pearsonr(x_pre.flatten(), x_true.flatten())
            r2 = r2_metric(x_true, x_pre)

            for k in range(outputs.shape[0]):
                prediction = torch.squeeze(outputs[k]).cpu().numpy()
                _image = torch.squeeze(image_batch[k]).cpu().numpy()
                _label = torch.squeeze(label_batch[k]).cpu().numpy()
                # image(1,128,128)

                # imageio.imwrite(f'{save_path}/{i_batch}_{k}_prediction.jpg', prediction)
                # imageio.imwrite(f'{save_path}/{i_batch}_{k}_image.jpg', _image)
                # imageio.imwrite(f'{save_path}/{i_batch}_{k}_label.jpg', _label)
                # save_image(prediction, f'{save_path}/{i_batch}_{k}_prediction.png')
                # save_image(_image, f'{save_path}/{i_batch}_{k}_image.png')
                # save_image(_label, f'{save_path}/{i_batch}_{k}_label.png')

                tifffile.imwrite(f'{save_path}/{i_batch}_{k}_prediction.tif', data=prediction)
                tifffile.imwrite(f'{save_path}/{i_batch}_{k}_image.tif', data=_image)
                tifffile.imwrite(f'{save_path}/{i_batch}_{k}_label.tif', data=_label)

            iter_num = iter_num + 1

            # Save csv.file
            meta_row = dict.fromkeys(DF_NAMES)
            meta_row['NRMSE'] = nrmse
            meta_row['SSIM'] = ssim_val
            meta_row['PCC'] = pcc
            meta_row['Dice'] = dice
            meta_row['mIOU'] = mIoU
            meta_row['r2'] = r2
            frames_meta.loc[iter_num - 1] = meta_row

            # record
            # loss_bank.append(loss.item())
            loss_bank.append(mse)
            nrmse_bank.append(nrmse)
            pcc_bank.append(pcc)
            dice_bank.append(dice)
            iou_bank.append(mIoU)
            ssim_bank.append(ssim_val)
            r2_bank.append(r2)

            logging.info('iteration %d : Loss: %.4f,NRMSE: %f,ssim: %f,PCC: %f,dice: %f,mIoU: %f,R2: %f' % (
                iter_num, loss.item(), nrmse, ssim_val, pcc, dice, mIoU, r2))

        frames_meta_filename = os.path.join("../predictions", args.model_path,
                                            f"inference_{args.model_name}_{args.day_time}.csv")
        frames_meta.to_csv(frames_meta_filename, sep=',')

        print('MSELoss: {:.4f}, NRMSE: {:.4f}, Dice: {:.4f}, IoU: {:.4f}, ssim:{:.4f}, PCC:{:.4f}, R2:{:.4f}'.
              format(np.mean(loss_bank), np.mean(nrmse_bank), np.mean(dice_bank), np.mean(iou_bank),
                     np.mean(ssim_bank), np.mean(pcc_bank), np.mean(r2_bank)))

        return np.mean(loss_bank)


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
            'volume_path': '/home/dataset/npy_1024/test',
            'label_dir': '/home/dataset/npy_256/test_label',
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
    size = (512, 512)
    # channels = (3, 32, 64, 128)
    # is_residual = True
    # bias = True
    # heads = 4
    # size = (128, 128)

    model_path = f'res_{is_residual}_head_{heads}_ch_{channels[-1]}'

    device = get_device()
    print(device, torch.cuda.device_count())

    args.model_path = model_path
    args.device = device
    args.size = size

    # name the same snapshot defined in train script!
    args.model_name = "UTransform"
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, args.model_path)

    # model = TransformerUNetParallel(channels, heads, size[0], is_residual, bias)
    model = TransformerUNet(channels, heads, is_residual, bias)

    # load checkpoint
    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', f'{args.model_name}-' + str(102))
    print('snapshot', snapshot, os.path.exists(snapshot))

    model.load_state_dict(torch.load(snapshot,map_location='cpu'), strict=False)
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

    # if args.n_gpu > 1:
    #     model = nn.DataParallel(model)

    test(args, model)
