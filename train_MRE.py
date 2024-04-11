import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
import importlib
import shutil
import time
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, confusion_matrix

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "datasets"))
sys.path.append(os.path.join(root_dir, "models"))
sys.path.append(os.path.join(root_dir, "optim"))

from datasets.dataset import data_loader
from build_model import build_model
from datasets.metric import *
from models.optimize import *

print("PyTorch Version: ", torch.__version__)


def FrobeniusNorm(input):  # [b,c,h,w]
    b, c, h, w = input.size()
    triu = torch.eye(h).cuda()
    triu = triu.unsqueeze(0).unsqueeze(0)
    triu = triu.repeat(b, c, 1, 1)

    x = torch.matmul(input, input.transpose(-2, -1))
    tr = torch.mul(x, triu)
    y = torch.sum(tr)
    return y


def print_table(data):
    col_width = [max(len(item) for item in col) for col in data]
    for row_idx in range(len(data[0])):
        for col_idx, col in enumerate(data):
            item = col[row_idx]
            align = '<' if not col_idx == 0 else '>'
            print(('{:' + align + str(col_width[col_idx]) + '}').format(item), end=" ")
        print()


def gmm_loss(label, prd, mu_f, mu_b, std_f, std_b, f_k):
    b_k = 1 - f_k

    f_likelihood = - f_k * (
            torch.log(np.sqrt(2 * 3.14) * std_f) + torch.pow((prd - mu_f), 2) / (2 * torch.pow(std_f, 2)) + 1e-10)
    b_likelihood = - b_k * (
            torch.log(np.sqrt(2 * 3.14) * std_b) + torch.pow((prd - mu_b), 2) / (2 * torch.pow(std_b, 2)) + 1e-10)
    likelihood = f_likelihood + b_likelihood
    loss = torch.mean(torch.pow(label - torch.exp(likelihood), 2))
    return loss


def random_crop(images, labels):
    now_size = images.shape[2]
    aim_size = now_size / 2
    trans = transforms.Compose([transforms.RandomCrop(aim_size)])
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
    cropped_labels.append(labels1.contiguous())
    cropped_imgs.append(images2)
    cropped_labels.append(labels2.contiguous())
    cropped_imgs.append(images3)
    cropped_labels.append(labels3.contiguous())
    cropped_imgs.append(images4)
    cropped_labels.append(labels4.contiguous())
    return cropped_imgs, cropped_labels




def train_one_epoch(epoch, loss_type, total_steps, dataloader, model,
                    device, criterion, optimizer, display_iter, log_file, writer):
    model.train()

    smooth_loss = 0.0
    current_step = 0
    t0 = 0.0

    for inputs in dataloader:

        t1 = time.time()

        images = inputs['image'].to(device)
        labels = inputs['mask'].to(device)

        cropped_imgs, cropped_labels = pre_process_il(images, labels)
        # forward pass
        pred = model(images, cropped_imgs)

        # compute loss
        loss = loss_compute(criterion, pred, labels, cropped_labels, loss_type)
        # predictions
        t0 += (time.time() - t1)

        total_steps += 1
        current_step += 1
        smooth_loss += loss.item()

        # back-propagate when training
        optimizer.zero_grad()

        lr_now = optimizer.state_dict()['param_groups'][0]['lr']
        # lr_update = update_learning_rate(optimizer, epoch, lr, step=lr_decay)

        loss.backward()
        optimizer.step()

        # log loss
        if total_steps % display_iter == 0:
            smooth_loss = smooth_loss / current_step
            message = "Epoch: %d Step: %d LR: %.6f Loss: %.4f Runtime: %.2fs/%diters." % (
                epoch + 1, total_steps, lr_now, smooth_loss, t0, display_iter)
            print("==> %s" % (message))
            writer.add_scalar('loss', smooth_loss, total_steps)
            writer.add_scalar('lr', lr_now, total_steps)
            with open(log_file, "a+") as fid:
                fid.write('%s\n' % message)

            t0 = 0.0
            current_step = 0
            smooth_loss = 0.0

    return total_steps


def eval_one_epoch(epoch, loss_type, threshold, dataloader, model, device, log_file, writer, mode='test'):
    with torch.no_grad():
        model.eval()

        total_iou = 0.0
        total_f1 = 0.0
        total_acc = 0.0
        total_img = 0

        for inputs in dataloader:
            images = inputs['image'].to(device)
            labels = inputs['mask']

            total_img += len(images)
            cropped_images, cropped_labels = pre_process_il(images, labels)
            outputs = model(images, cropped_images)

            if "single" in loss_type:
                y_scores = outputs.cpu().numpy()
            else:
                y_scores = outputs[0].cpu().numpy()
            y_true = labels.cpu().numpy()
            y_labels = np.zeros_like(y_true)
            y_preds = np.zeros_like(y_scores)
            y_preds[y_scores > threshold] = 1
            y_labels[y_true > 0.01] = 1

            y_preds = y_preds.flatten()
            y_labels = y_labels.flatten()

            # total_distance = 0.0

            confusion = confusion_matrix(y_labels, y_preds)
            tp = float(confusion[1, 1])
            fn = float(confusion[1, 0])
            fp = float(confusion[0, 1])
            tn = float(confusion[0, 0])

            iou = tp / (tp + fn + fp + 1e-9)
            val_acc = (tp + tn) / (tp + fn + fp + tn)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            precision = tp / (tp + fp)
            f1 = 2 * sensitivity * precision / (sensitivity + precision + 1e-9)

            # metric
            total_acc = val_acc
            total_iou = iou
            total_f1 = f1

        epoch_iou = total_iou
        epoch_f1 = total_f1
        epoch_acc = total_acc
        if mode == 'test':
            message = "total Threshold: {:.3f} =====> Evaluation Iou: {:.4f}; F1_score: {:.4f}; Acc: {:.4f}".format(
                threshold, epoch_iou, epoch_f1, epoch_acc)
            print("==> %s" % (message))
            with open(log_file, "a+") as fid:
                fid.write('%s\n' % message)
        writer.add_scalar(f'{mode}_ious', epoch_iou, epoch)

    return epoch_acc, epoch_iou, epoch_f1


def train_eval_model(opts):
    # parse model configuration
    num_epochs = opts["num_epochs"]
    train_batch_size = opts["train_batch_size"]
    val_batch_size = opts["eval_batch_size"]
    dataset_type = opts["dataset_type"]
    model_type = opts["model_type"]

    opti_mode = opts["optimizer"]
    loss_criterion = opts["loss_criterion"]
    loss_type = opts["loss_type"]
    wd = opts["weight_decay"]
    lr = opts["lr"]

    gpus = opts["gpu_list"].split(',')
    os.environ['CUDA_VISIBLE_DEVICE'] = opts["gpu_list"]
    train_dir = opts["log_dir"]
    writer = SummaryWriter(train_dir + '/log')

    train_data_dir = opts["train_data_dir"]
    eval_data_dir = opts["eval_data_dir"]
    pretrained = opts["pretrained_model"]
    resume = opts["resume"]
    display_iter = opts["display_iter"]
    save_epoch = opts["save_every_epoch"]

    # backup train configs
    log_file = os.path.join(train_dir, "log_file.txt")
    os.makedirs(train_dir, exist_ok=True)
    model_dir = os.path.join(train_dir, "code_backup")
    os.makedirs(model_dir, exist_ok=True)

    if resume is None and os.path.exists(log_file): os.remove(log_file)
    shutil.copy("./models/pvt_MRE.py", os.path.join(model_dir, "pvt_MRE.py"))
    shutil.copy("train_MRE.py", os.path.join(model_dir, "train_MRE.py"))
    shutil.copy("./datasets/dataset.py", os.path.join(model_dir, "dataset.py"))

    ckt_dir = os.path.join(train_dir, "checkpoints")
    os.makedirs(ckt_dir, exist_ok=True)

    # format printing configs
    print("*" * 50)
    table_key = []
    table_value = []
    n = 0
    for key, value in opts.items():
        table_key.append(key)
        table_value.append(str(value))
        n += 1
    print_table([table_key, ["="] * n, table_value])

    # format gpu list
    gpu_list = []
    for str_id in gpus:
        id = int(str_id)
        gpu_list.append(id)

    # dataloader
    print("==> Create dataloader")
    dataloaders_dict = {
        "train": data_loader(train_data_dir, train_batch_size, dataset_type, is_train=True),
        "eval": data_loader(eval_data_dir, val_batch_size, dataset_type, is_train=False)}

    # define parameters of two networks

    print("==> Create network")
    model = build_model(opts)

    # loss layer
    criterion = create_criterion(criterion=loss_criterion)

    best_acc = 0.0
    best_iou = 0.0
    start_epoch = 0

    # load pretrained model
    if pretrained is not None and os.path.isfile(pretrained):
        print("==> Train from model '{}'".format(pretrained))
        checkpoint_gan = torch.load(pretrained)
        model.load_state_dict(checkpoint_gan['model_state_dict'])
        print("==> Loaded checkpoint '{}')".format(pretrained))
        for param in model.parameters():
            param.requires_grad = False

    # resume training
    elif resume is not None and os.path.isfile(resume):
        print("==> Resume from checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        print("==> Loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch'] + 1))

    # train from scratch
    else:
        print("==> Train from initial or random state.")

    # define mutiple-gpu mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.cuda()
    model = nn.DataParallel(model)

    # print learnable parameters
    print("==> List learnable parameters")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t{}, size {}".format(name, param.size()))
    params_to_update = [{'params': model.parameters()}]

    # define optimizer
    print("==> Create optimizer")
    optimizer = create_optimizer(params_to_update, opti_mode, lr=lr, momentum=0.9, wd=wd)
    if resume is not None and os.path.isfile(resume): optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = get_scheduler(optimizer, opts)
    # start training
    since = time.time()

    # Each epoch has a training and validation phase
    print("==> Start training")
    total_steps = 0
    threshold = opts["threshold"]
    epochs = []
    ious = []
    for epoch in range(start_epoch, num_epochs):
        epochs.append(epoch)
        print('-' * 50)
        print("==> Epoch {}/{}".format(epoch + 1, num_epochs))

        total_steps = train_one_epoch(epoch, loss_type, total_steps,
                                      dataloaders_dict['train'],
                                      model, device,
                                      criterion, optimizer, display_iter, log_file, writer)

        scheduler.step()
        epoch_acc, epoch_iou, epoch_f1 = eval_one_epoch(epoch, loss_type, threshold, dataloaders_dict['eval'],
                                                        model, device, log_file, writer, mode='test')
        ious.append(epoch_iou)
        if best_iou < epoch_iou and epoch >= int(num_epochs - 30):
            best_acc = epoch_acc
            best_iou = epoch_iou
            torch.save({'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': best_acc},
                       os.path.join(ckt_dir, "best.pth"))

        if (epoch + 1) % save_epoch == 0 and (epoch + 1) >= int(num_epochs - 6):
            torch.save({'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': epoch_acc},
                       os.path.join(ckt_dir, "checkpoints_" + str(epoch + 1) + ".pth"))

    time_elapsed = time.time() - since
    time_message = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    print(time_message)
    with open(log_file, "a+") as fid:
        fid.write('%s\n' % time_message)
    print('==> Best val Acc: {:4f}; Iou: {:4f}'.format(best_acc, best_iou))


if __name__ == '__main__':
    date = '20240411'
    # the following parameters need to be set

    # the model type
    model_choice = ['PVT_MRE', 'UNet_MRE']
    # the dataset type
    dataset_list = ['er', 'retina', 'mito', 'stare']
    # the encoder type
    encoder_choice = ['unet','pvt_tiny', 'pvt_small', 'pvt_base', 'pvt_large']
    # the loss type
    loss_choice = ['single', 'up_sampling', 'multi_layer', 'hierarchical_fusing']

    txt_choice = ['test_drive.txt', 'train_drive.txt', 'train_mito.txt', 'test_mito.txt', 'train_er.txt',
                  'test_er.txt', 'test_stare.txt', 'train_stare.txt']

    opts = dict()
    opts['dataset_type'] = 'er'
    opts['model_type'] = 'UNet_MRE'
    opts['encoder_type'] = 'unet'
    opts["num_channels"] = 1
    opts["num_classes"] = 1
    opts["loss_criterion"] = "iou" # the basic type of loss function
    opts["loss_type"] = "hierarchical_fusing"
    opts["num_epochs"] = 100
    opts["train_data_dir"] = "./dataset_txts/train_er.txt"
    opts["eval_data_dir"] = "./dataset_txts/test_er.txt"
    opts["train_batch_size"] = 16
    opts["eval_batch_size"] = 38
    opts["lr"] = 0.0005
    opts["warm_up_epochs"] = 20
    opts["lr_milestones"] = [80]
    opts["threshold"] = 0.3
    opts["optimizer"] = "Adam"
    opts["lr_policy"] = "warm_up_multi_step"
    opts["weight_decay"] = 0.05
    opts["gpu_list"] = "0,1,2,3"
    log_dir = "./train_logs/" + str(opts["dataset_type"])+ '_' + str(
        opts["model_type"]) + '_' + str(opts["encoder_type"]) + '_' + opts["loss_type"] + '_' + opts[
                  "lr_policy"] + '_' + str(
        opts["train_batch_size"]) + '_' + str(opts["lr"]) + '_' + str(
        opts["num_epochs"]) + '_' + str(opts["threshold"]) + '_' + str(
        opts["warm_up_epochs"]) + '_' + str(opts["lr_milestones"][0]) + '_' + date + '_' + str(
        opts["weight_decay"])
    opts["log_dir"] = log_dir
    opts["pretrained_model"] = None
    opts["resume"] = None
    opts["display_iter"] = 10
    opts["save_every_epoch"] = 5
    opts["vis"] = False

    train_eval_model(opts)



