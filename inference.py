from __future__ import print_function

from datasets.dataset import data_loader

from build_model import build_model
from datasets.metric import *
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, confusion_matrix

print("PyTorch Version: ", torch.__version__)

'''
evaluation
'''

def random_crop(images, labels):
    now_size1 = images.shape[2]
    now_size2 = images.shape[3]
    aim_size1 = int(now_size1 / 2)
    aim_size2 = int(now_size2 / 2)
    trans = transforms.Compose([transforms.RandomCrop((aim_size1, aim_size2))])
    seed = torch.random.seed()
    torch.random.manual_seed(seed)
    cropped_img = trans(images)
    torch.random.manual_seed(seed)
    cropped_label = trans(labels)
    return cropped_img, cropped_label


def pre_process_il(images, labels):
    # level crop
    cropped_imgs = []
    cropped_labels = []

    images1, labels1 = random_crop(images, labels)
    images2, labels2 = random_crop(images1, labels1)
    images3, labels3 = random_crop(images2, labels2)
    images4, labels4 = random_crop(images3, labels3)

    cropped_imgs.append(images1)
    cropped_labels.append(labels1)
    cropped_imgs.append(images2)
    cropped_labels.append(labels2)
    cropped_imgs.append(images3)
    cropped_labels.append(labels3)
    cropped_imgs.append(images4)
    cropped_labels.append(labels4)
    return cropped_imgs, cropped_labels


def eval_model(opts, index):
    val_batch_size = opts["eval_batch_size"]
    dataset_type = opts['dataset_type']
    load_epoch = opts['load_epoch']
    loss_type = opts['loss_type']
    gpus = opts["gpu_list"].split(',')
    gpu_list = []
    for str_id in gpus:
        id = int(str_id)
        gpu_list.append(id)
    os.environ['CUDA_VISIBLE_DEVICE'] = opts["gpu_list"]

    eval_data_dir = opts["eval_data_dir"]
    dataset_name = os.path.split(eval_data_dir)[-1].split('.')[0]

    train_dir = opts["train_dir"]

    model_score_dir = os.path.join(str(os.path.split(train_dir)[0]),
                                   'predict_score/' + dataset_name + '_' + str(load_epoch))
    if not os.path.exists(model_score_dir): os.makedirs(model_score_dir)

    # dataloader
    print("==> Create dataloader")
    dataloader = data_loader(eval_data_dir, val_batch_size, dataset_type, is_train=False)

    # define network
    print("==> Create network")
    model = build_model(opts)

    # load trained model
    pretrain_model = os.path.join(train_dir, str(load_epoch) + ".pth")
    # print(pretrain_model)
    # pretrain_model = os.path.join(train_dir, "checkpoints_" + str(load_epoch) + ".pth")

    if os.path.isfile(pretrain_model):
        c_checkpoint = torch.load(pretrain_model)
        model.load_state_dict(c_checkpoint["model_state_dict"])
        print("==> Loaded pretrianed model checkpoint '{}'.".format(pretrain_model))
    else:
        print("==> No trained model.")
        return 0

    # set model to gpu mode
    print("==> Set to GPU mode")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=gpu_list)
    thresholds = np.arange(0.3, 0.4, 0.05)
    best_iou = 0.00
    # enable evaluation mode
    with torch.no_grad():
        model.eval()
        for threshold in thresholds:
            total_img = 0
            for inputs in dataloader:
                images = inputs["image"].cuda()
                labels = inputs['mask']

                cropped_imgs, cropped_labels = pre_process_il(images, labels)
                img_name = inputs['ID']
                total_img += len(images)

                # unet
                p_seg = model(images, cropped_imgs)

                if "single" not in loss_type:
                    pred_save = p_seg[0]
                    pred_compute = p_seg[0]
                else:
                    pred_save = p_seg
                    pred_compute = p_seg

                for i in range(len(images)):
                        now_dir = model_score_dir + '_' + str(index)
                        os.makedirs(now_dir, exist_ok=True)
                        np.save(os.path.join(now_dir, img_name[i].split('.')[0] + '.npy'),
                                pred_save[i][0].cpu().numpy().astype(np.float32))
                        cv2.imwrite(os.path.join(now_dir, img_name[i].split('.')[0] + '.tif'),
                                    pred_save[i][0].cpu().numpy().astype(np.float32))
                y_scores = pred_compute.cpu().numpy().flatten()
                y_pred = y_scores > threshold
                y_true = labels.numpy().flatten()

                # total_distance = 0.0

                confusion = confusion_matrix(y_true, y_pred)
                tp = float(confusion[1, 1])
                fn = float(confusion[1, 0])
                fp = float(confusion[0, 1])
                tn = float(confusion[0, 0])

                val_acc = (tp + tn) / (tp + fn + fp + tn)
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                precision = tp / (tp + fp)
                f1 = 2 * sensitivity * precision / (sensitivity + precision)
                iou = tp / (tp + fn + fp)
                auc = roc_auc_score(y_true, y_scores)

            epoch_iou = iou
            if epoch_iou > best_iou:
                best_iou = epoch_iou
            epoch_f1 = f1
            epoch_acc = val_acc
            epoch_auc = auc
            epoch_sen = sensitivity
            epoch_spec = specificity
            message = "inference  =====> Evaluation  ACC: {:.4f}; IOU: {:.4f}; F1_score: {:.4f}; Auc: {:.4f} ;Sen: {:.4f}; Spec: {:.4f}; threshold: {:.4f}".format(
                epoch_acc,
                epoch_iou,
                epoch_f1, epoch_auc, epoch_sen, epoch_spec, threshold)
            print("==> %s" % (message))

        print("validation image number {}".format(total_img))
    return best_iou


if __name__ == "__main__":
    model_choice = ['PVT_MRE', 'UNet_MRE']
    encoder_choice = ['unet', 'pvt_tiny', 'pvt_small', 'pvt_base', 'pvt_large']
    loss_choice = ['single', 'up_sampling', 'multi_layer', 'hierarchical_fusing']
    dataset_list = ['er', 'retina', 'mito', 'stare']
    txt_choice = ['test_drive.txt', 'test_mito.txt', 'test_er.txt', 'test_stare.txt']

    size_choice = [80, 10, 38, 40]
    opts = dict()
    opts['dataset_type'] = 'er'
    opts["eval_batch_size"] = 38
    opts["gpu_list"] = "0,1,2,3"
    opts[
        "train_dir"] = "/mnt/data1/hjx_code/MHCNet_code/train_logs/er_train_MHCnet_Hierarchical_Fusing_Loss_iouloss_16_0.17_30_0.3_100_20231229_0.0005_warmup_random_crop/checkpoints/"
    opts["eval_data_dir"] = "/mnt/data1/hjx_data/dataset_txts/test_er.txt"
    opts['model_type'] = 'PVT_MRE'
    opts['encoder_type'] = 'pvt_tiny'
    opts["num_channels"] = 1
    opts["num_classes"] = 1
    opts["loss_type"] = "hierarchical_fusing"
    opts["load_epoch"] = 'best'

    best_iou = 0.0
    for i in range(10):
        print('************now time//' + str(i))
        now_iou = eval_model(opts, i)
        if now_iou > best_iou:
            best_iou = now_iou
    print('final best iou =' + str(best_iou))
