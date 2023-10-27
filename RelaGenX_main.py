# import argparse
import os
import time
import shutil
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from dataset import *
from models import VideoModel
from loss import *
from opts_hmdb_ucf_shot1 import parser
from utils.utils import randSelectBatch
import math
from torch.utils.data import WeightedRandomSampler
from colorama import init
from colorama import Fore, Back, Style
import numpy as np
from tensorboardX import SummaryWriter
from info_nce import InfoNCE
from pytorch_metric_learning import miners, losses

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.cuda.manual_seed(42)
#torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mem_bank = [[] for i in range(12)]
init(autoreset=True)

best_prec1 = 0
gpu_count = torch.cuda.device_count()
class CustomImageDataset(Dataset):
    def __init__(self, feature, annotation, transform=None, target_transform=None):

        self.feature = feature #.view(-1,34)
        #print(self.feature.shape)
        #sys-exit()
        self.annotations = annotation
    def __len__(self):
        return len(self.annotations)*5
    def __getitem__(self, idx):
        if idx >= len(self.annotations):
            idx = idx%len(self.annotations)
        data_ancher = self.feature[idx]
        perm = np.random.permutation(data_ancher.shape[0])
        data_pos = data_ancher[perm]            
        index_neg = np.random.randint(len(self.annotations))
        record_neg = self.feature[index_neg]
        return data_ancher, self.annotations[idx], data_pos, record_neg 
def main():
	
    global args, best_prec1, writer
    args = parser.parse_args()

    print(Fore.GREEN + 'Baseline:', args.baseline_type)
    print(Fore.GREEN + 'Frame aggregation method:', args.frame_aggregation)

    print(Fore.GREEN + 'target data usage:', args.use_target)
    if args.use_target == 'none':
        print(Fore.GREEN + 'no Domain Adaptation')
    else:
        if args.dis_DA != 'none':
            print(Fore.GREEN + 'Apply the discrepancy-based Domain Adaptation approach:', args.dis_DA)
            if len(args.place_dis) != args.add_fc + 2:
                raise ValueError(Back.RED + 'len(place_dis) should be equal to add_fc + 2')

        if args.adv_DA != 'none':
            print(Fore.GREEN + 'Apply the adversarial-based Domain Adaptation approach:', args.adv_DA)

        if args.use_bn != 'none':
            print(Fore.GREEN + 'Apply the adaptive normalization approach:', args.use_bn)

    # determine the categories
    #class_names = [line.strip().split(' ', 1)[1] for line in open(args.class_file)]
    num_class = 12#len(class_names)

    #=== check the folder existence ===#
    path_exp = args.exp_path + args.modality + '/'
    if not os.path.isdir(path_exp):
        os.makedirs(path_exp)

    if args.tensorboard:
        writer = SummaryWriter(path_exp + '/tensorboard')  # for tensorboardX

    #=== initialize the model ===#
    print(Fore.CYAN + 'preparing the model......')
    model = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
                train_segments=args.num_segments, val_segments=args.val_segments, 
                base_model=args.arch, path_pretrained=args.pretrained,
                add_fc=args.add_fc, fc_dim = args.fc_dim,
                dropout_i=args.dropout_i, dropout_v=args.dropout_v, partial_bn=not args.no_partialbn,
                use_bn=args.use_bn if args.use_target != 'none' else 'none', ens_DA=args.ens_DA if args.use_target != 'none' else 'none',
                n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
                use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
                verbose=args.verbose, share_params=args.share_params)

    model = torch.nn.DataParallel(model, args.gpus).cuda()

    if args.optimizer == 'SGD':
        print(Fore.YELLOW + 'using SGD')
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'Adam':
        print(Fore.YELLOW + 'using Adam')
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        print(Back.RED + 'optimizer not support or specified!!!')
        exit()

    #=== check point ===#
    start_epoch = 1
    print(Fore.CYAN + 'checking the checkpoint......')
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch'])))
            if args.resume_hp:
                print("=> loaded checkpoint hyper-parameters")
                optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print(Back.RED + "=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #--- open log files ---#
    if not args.evaluate:
        if args.resume:
            train_file = open(path_exp + 'train.log', 'a')
            train_short_file = open(path_exp + 'train_short.log', 'a')
            val_file = open(path_exp + 'val.log', 'a')
            val_short_file = open(path_exp + 'val_short.log', 'a')
            train_file.write('========== start: ' + str(start_epoch) + '\n')  # separation line
            train_short_file.write('========== start: ' + str(start_epoch) + '\n')
            val_file.write('========== start: ' + str(start_epoch) + '\n')
            val_short_file.write('========== start: ' + str(start_epoch) + '\n')
        else:
            train_short_file = open(path_exp + 'train_short.log', 'w')
            val_short_file = open(path_exp + 'val_short.log', 'w')
            train_file = open(path_exp + 'train.log', 'w')
            val_file = open(path_exp + 'val.log', 'w')

        val_best_file = open(args.save_best_log, 'a')

    else:
        test_short_file = open(path_exp + 'test_short.log', 'w')
        test_file = open(path_exp + 'test.log', 'w')

    #=== Data loading ===#
    print(Fore.CYAN + 'loading data......')

    if args.use_opencv:
        print("use opencv functions")

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
        data_length = 5
    shot=1
    ref_round=0 #/pfs/work7/workspace/scratch/fy2374-train/acmmm_and_aaai/ZDDA/DANN/dataset/new_ucfhmdb
    ref_lists=['/pfs/work7/workspace/scratch/fy2374-train/acmmm_and_aaai/ZDDA/new_ucfhmdb/ref_hmdb2ucf_'+str(shot)+'shot_split'+str(i)+'.txt' for i in range(12)]
    ref_path= ref_lists[ref_round]

    args.train_source_list = '/pfs/work7/workspace/scratch/fy2374-train/acmmm_and_aaai/ZDDA/TranSVAE/dataset/hmdb51/list/I3Dpretrain/list_train_hmdb51-ucf101_I3Dpretrain.txt'
    args.train_target_list = ref_path
    args.val_list = '/pfs/work7/workspace/scratch/fy2374-train/acmmm_and_aaai/ZDDA/TranSVAE/dataset/ucf101/list/I3Dpretrain/list_val_hmdb51-ucf101_I3Dpretrain.txt'

    # calculate the number of videos to load for training in each list ==> make sure the iteration # of source & target are same
    num_source = sum(1 for i in open(args.train_source_list))
    num_target = sum(1 for i in open(args.train_target_list))
    num_val = sum(1 for i in open(args.val_list))


    num_iter_source = num_source / args.batch_size[0]
    num_iter_target = num_target / args.batch_size[1]
    num_max_iter = max(num_iter_source, num_iter_target)
    num_source_train = round(num_max_iter*args.batch_size[0]) if args.copy_list[0] == 'Y' else num_source
    num_target_train = round(num_max_iter*args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target

    # calculate the weight for each class
    class_id_list = [int(line.strip().split(' ')[2]) for line in open(args.train_source_list)]
    class_id, class_data_counts = np.unique(np.array(class_id_list), return_counts=True)
    class_freq = (class_data_counts / class_data_counts.sum()).tolist()

    weight_source_class = torch.ones(num_class).cuda()
    weight_domain_loss = torch.Tensor([1, 1]).cuda()

    if args.weighted_class_loss == 'Y':
        weight_source_class = 1 / torch.Tensor(class_freq).cuda()

    if args.weighted_class_loss_DA == 'Y':
        weight_domain_loss = torch.Tensor([1/num_source_train, 1/num_target_train]).cuda()

    # data loading (always need to load the testing data)
    val_segments = args.val_segments if args.val_segments > 0 else args.num_segments
    val_set = TSNDataSet("/pfs/work7/workspace/scratch/fy2374-train/acmmm_and_aaai/ucf-101_features_new/",
                        "/pfs/work7/workspace/scratch/fy2374-train/acmmm_and_aaai/ucf-101_features_new/",
                            "/pfs/work7/workspace/scratch/fy2374-train/acmmm_and_aaai/ZDDA/TranSVAE/dataset/ucf101/list/I3Dpretrain/list_val_hmdb51-ucf101_I3Dpretrain.txt",
                            num_dataload=num_val,
                            num_segments=val_segments,
                            new_length=1, modality='RGB',
                            image_tmpl="img_{:05d}.t7",
                            random_shift=False,
                            test_mode=True
                            )
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size[2], shuffle=False,
                                                num_workers=args.workers, pin_memory=True)

    if not args.evaluate:
        source_set = TSNDataSet("/pfs/work7/workspace/scratch/fy2374-train/acmmm_and_aaai/hmdb_ZDDA/hmdb_rgb/RGB-feature_i3d/",
                                "/pfs/work7/workspace/scratch/fy2374-train/acmmm_and_aaai/hmdb_ZDDA/hmdb_rgb/RGB-feature_i3d/",
                                "/pfs/work7/workspace/scratch/fy2374-train/acmmm_and_aaai/ZDDA/TranSVAE/dataset/hmdb51/list/I3Dpretrain/list_train_hmdb51-ucf101_I3Dpretrain.txt",
                                num_dataload=num_source,
                                num_segments=12,
                                new_length=1,
                                modality='RGB',
                                image_tmpl="img_{:05d}.t7",
                                random_shift=False,
                                test_mode=True,
                                triple=False
                                )
        source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
        source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size[0], shuffle=False, sampler=source_sampler, num_workers=args.workers, pin_memory=True)

        target_set = TSNDataSet("/pfs/work7/workspace/scratch/fy2374-train/acmmm_and_aaai/ucf-101_features_new/",
                                "/pfs/work7/workspace/scratch/fy2374-train/acmmm_and_aaai/ucf-101_features_new/",
                                ref_path,
                                num_dataload=num_source, num_segments=12,
                                new_length=1,
                                modality='RGB',
                                image_tmpl="img_{:05d}.t7",
                                random_shift=False,
                                test_mode=True,
                                triple=False
                                )
        target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
        target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size[1], shuffle=False, sampler=target_sampler, num_workers=args.workers, pin_memory=True)

    import pickle as pkl
    f = open('generated_new_data_hmdb_ucf_shot1.pkl', 'rb')
    auxiliary_data = pkl.load(f)
    f.close()

    auxiliary_label = np.concatenate([torch.ones(200)*j for j in range(12)])
    auxiliary_dataset = CustomImageDataset(auxiliary_data, auxiliary_label)
    aux_sampler = torch.utils.data.sampler.RandomSampler(auxiliary_dataset)
    aux_loader = torch.utils.data.DataLoader(auxiliary_dataset, batch_size=args.batch_size[0], sampler=aux_sampler, num_workers=2, pin_memory=True)

    # --- Optimizer ---#
    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss(weight=weight_source_class).cuda()
        criterion_domain = torch.nn.CrossEntropyLoss(weight=weight_domain_loss).cuda()
    else:
        raise ValueError("Unknown loss type")

    if args.evaluate:
        print(Fore.CYAN + 'evaluation only......')
        prec1 = validate(val_loader, model, criterion, num_class, 0, test_file)
        test_short_file.write('%.3f\n' % prec1)
        return

    #=== Training ===#
    start_train = time.time()
    print(Fore.CYAN + 'start training......')
    beta = args.beta
    gamma = args.gamma
    mu = args.mu
    loss_c_current = 999 # random large number
    loss_c_previous = 999 # random large number

    attn_source_all = torch.Tensor()
    attn_target_all = torch.Tensor()

    for epoch in range(start_epoch, args.epochs+1):

        ## schedule for parameters
        alpha = 2 / (1 + math.exp(-1 * (epoch) / args.epochs)) - 1 if args.alpha < 0 else args.alpha

        ## schedule for learning rate
        if args.lr_adaptive == 'loss':
            adjust_learning_rate_loss(optimizer, args.lr_decay, loss_c_current, loss_c_previous, '>')
        elif args.lr_adaptive == 'none' and epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args.lr_decay)
        #prototypes = epoch_wise_prototype_calculation(model, target_loader, beta, mu)
        # train for one epoch
        loss_c, attn_epoch_source, attn_epoch_target = train(num_class, aux_loader, source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, train_file, train_short_file, alpha, beta, gamma, mu)
        
        if args.save_attention >= 0:
            attn_source_all = torch.cat((attn_source_all, attn_epoch_source.unsqueeze(0)))  # save the attention values
            attn_target_all = torch.cat((attn_target_all, attn_epoch_target.unsqueeze(0)))  # save the attention values

        # update the recorded loss_c
        loss_c_previous = loss_c_current
        loss_c_current = loss_c

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            prec1 = validate(val_loader, model, criterion, num_class, epoch, val_file)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            line_update = ' ==> updating the best accuracy' if is_best else ''
            line_best = "Best score {} vs current score {}".format(best_prec1, prec1) + line_update
            print(Fore.YELLOW + line_best)
            val_short_file.write('%.3f\n' % prec1)

            best_prec1 = max(prec1, best_prec1)

            if args.tensorboard:
                writer.add_text('Best_Accuracy', str(best_prec1), epoch)

            if args.save_model:
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'best_prec1': best_prec1,
                    'prec1': prec1,
                }, is_best, path_exp)

    end_train = time.time()
    print(Fore.CYAN + 'total training time:', end_train - start_train)
    val_best_file.write('%.3f\n' % best_prec1)

    # --- write the total time to log files ---#
    line_time = 'total time: {:.3f} '.format(end_train - start_train)
    if not args.evaluate:
        train_file.write(line_time)
        train_short_file.write(line_time)
        val_file.write(line_time)
        val_short_file.write(line_time)
    else:
        test_file.write(line_time)
        test_short_file.write(line_time)

    #--- close log files ---#
    if not args.evaluate:
        train_file.close()
        train_short_file.close()
        val_file.close()
        val_short_file.close()
    else:
        test_file.close()
        test_short_file.close()

    if args.tensorboard:
        writer.close()

    if args.save_attention >= 0:
        np.savetxt('attn_source_' + str(args.save_attention) + '.log', attn_source_all.cpu().detach().numpy(), fmt="%s")
        np.savetxt('attn_target_' + str(args.save_attention) + '.log', attn_target_all.cpu().detach().numpy(), fmt="%s")

    '''def epoch_wise_prototype_calculation(model, target_loader, beta, mu):
    class_num = 8
    prototypes = torch.zeros(8,256).cuda()
    counter = torch.zeros(8).cuda()
    for i, (source_data, source_label) in enumerate(target_loader):
        attn_source, out_source, out_source_2, pred_domain_source, feat_source, attn_target, out_target, out_target_2, pred_domain_target, feat_target = model(source_data, source_data, beta, mu, is_train=True, reverse=False)
        feat = feat_source[1]
        for i, k in enumerate(source_label):
            counter[k] += 1
            prototypes[k] += feat[i]
    return prototypes/counter.unsqueeze(-1).repeat(1,256)'''

class CenterLoss(nn.Module):

    def __init__(self, num_classes=8, feat_dim=256, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                    torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


def train(num_class,aux_loader, source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, log, log_short, alpha, beta, gamma, mu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_a = AverageMeter()  # adversarial loss
    losses_d = AverageMeter()  # discrepancy loss
    losses_e = AverageMeter()  # entropy loss
    losses_s = AverageMeter()  # ensemble loss
    losses_c = AverageMeter()  # classification loss
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    tpl_loss = CenterLoss().cuda()
    #miner = miners.MultiSimilarityMiner()
    #tpl_loss = losses.TripletMarginLoss().cuda()
    #kld = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean').cuda()
    info_nce_loss =  InfoNCE().cuda()
    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    data_loader = enumerate(zip(source_loader, target_loader))

    # step info
    start_steps = epoch * len(source_loader)
    total_steps = args.epochs * len(source_loader)

    # initialize the embedding
    if args.tensorboard:
        feat_source_display = None
        label_source_display = None
        label_source_domain_display = None

        feat_target_display = None
        label_target_display = None
        label_target_domain_display = None
    aux_iter = iter(aux_loader)
    attn_epoch_source = torch.Tensor()
    attn_epoch_target = torch.Tensor()
    for i, ((source_data, source_label, source_pos, source_neg),(target_data, target_label, target_pos, target_neg)) in data_loader:
        # setup hyperparameters
        p = float(i + start_steps) / total_steps
        beta_dann = 2. / (1. + np.exp(-10 * p)) - 1
        beta = [beta_dann if beta[i] < 0 else beta[i] for i in range(len(beta))] # replace the default beta if value < 0

        aux_data, aux_label, aux_pos, aux_neg = next(aux_iter)

        source_size_ori = source_data.size()  # original shape
        target_size_ori = target_data.size()  # original shape
        batch_source_ori = source_size_ori[0]
        batch_target_ori = target_size_ori[0]
        # add dummy tensors to keep the same batch size for each epoch (for the last epoch)
        if batch_source_ori < args.batch_size[0]:
            source_data_dummy = torch.zeros(args.batch_size[0] - batch_source_ori, source_size_ori[1], source_size_ori[2])
            source_data = torch.cat((source_data, source_data_dummy))
            source_data_dummy_pos = torch.zeros(args.batch_size[0] - batch_source_ori, source_size_ori[1], source_size_ori[2])
            source_pos = torch.cat((source_pos, source_data_dummy_pos))
            source_data_dummy_neg = torch.zeros(args.batch_size[0] - batch_source_ori, source_size_ori[1], source_size_ori[2])
            source_neg = torch.cat((source_neg, source_data_dummy_neg))
        if batch_target_ori < args.batch_size[1]:
            target_data_dummy = torch.zeros(args.batch_size[1] - batch_target_ori, target_size_ori[1], target_size_ori[2])
            target_data = torch.cat((target_data, target_data_dummy))
            target_data_dummy_pos = torch.zeros(args.batch_size[1] - batch_target_ori, target_size_ori[1], target_size_ori[2])
            target_pos = torch.cat((target_pos, target_data_dummy_pos))
            target_data_dummy_neg = torch.zeros(args.batch_size[1] - batch_target_ori, target_size_ori[1], target_size_ori[2])
            target_neg = torch.cat((target_neg, target_data_dummy_neg))
        # add dummy tensors to make sure batch size can be divided by gpu #
        if source_data.size(0) % gpu_count != 0:
            source_data_dummy = torch.zeros(gpu_count - source_data.size(0) % gpu_count, source_data.size(1), source_data.size(2))
            source_data = torch.cat((source_data, source_data_dummy))
            source_data_dummy_pos = torch.zeros(gpu_count - source_data.size(0) % gpu_count, source_data.size(1), source_data.size(2))
            source_pos = torch.cat((source_pos, source_data_dummy_pos))
            source_data_dummy_neg = torch.zeros(gpu_count - source_data.size(0) % gpu_count, source_data.size(1), source_data.size(2))
            source_neg = torch.cat((source_neg, source_data_dummy_neg))
        if target_data.size(0) % gpu_count != 0:
            target_data_dummy = torch.zeros(gpu_count - target_data.size(0) % gpu_count, target_data.size(1), target_data.size(2))
            target_data = torch.cat((target_data, target_data_dummy))
            target_data_dummy_pos = torch.zeros(gpu_count - target_data.size(0) % gpu_count, target_data.size(1), target_data.size(2))
            target_pos = torch.cat((target_pos, target_data_dummy_pos))
            target_data_dummy_neg = torch.zeros(gpu_count - target_data.size(0) % gpu_count, target_data.size(1), target_data.size(2))
            target_neg = torch.cat((target_neg, target_data_dummy_neg))
        # measure data loading time
        data_time.update(time.time() - end)

        source_label = source_label.cuda(non_blocking=True) # pytorch 0.4.X
        target_label = target_label.cuda(non_blocking=True) # pytorch 0.4.X
        aux_data = torch.Tensor(aux_data.numpy().astype(np.float32)).float().cuda()
        aux_label = torch.Tensor(aux_label).cuda().long()
        aux_pos = torch.Tensor(aux_pos.numpy().astype(np.float32)).float().cuda()
        aux_neg = torch.Tensor(aux_neg.numpy().astype(np.float32)).float().cuda()


        source_data = source_data.cuda()
        target_data = target_data.cuda()
        source_pos = source_pos.cuda()
        target_pos = target_pos.cuda()
        source_neg = source_neg.cuda()
        target_neg = target_neg.cuda()
        aux = torch.cat([aux_pos, aux_neg], 0)
        if args.baseline_type == 'frame':
            source_label_frame = source_label.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames
            target_label_frame = target_label.unsqueeze(1).repeat(1, args.num_segments).view(-1)
        
        label_source = source_label_frame if args.baseline_type == 'frame' else source_label  # determine the label for calculating the loss function
        label_target = target_label_frame if args.baseline_type == 'frame' else target_label

        #====== forward pass data ======#
        attn_source, out_source, out_source_2, pred_domain_source, feat_source, attn_target, out_target, out_target_2, pred_domain_target, feat_target = model(source_data, target_data, beta, mu, is_train=True, reverse=False)
        attn_source_pos, out_source_pos, out_source_2_pos, pred_domain_source_pos, feat_source_pos, attn_target_pos, out_target_pos, out_target_2_pos, pred_domain_target_pos, feat_target_pos = model(source_pos, target_pos, beta, mu, is_train=True, reverse=False)
        attn_source_neg, out_source_neg, out_source_2_neg, pred_domain_source_neg, feat_source_neg, attn_target_neg, out_target_neg, out_target_2_neg, pred_domain_target_neg, feat_target_neg = model(source_neg, target_neg, beta, mu, is_train=True, reverse=False)
        attn_aux, out_aux, out_aux_2, pred_domain_aux, feat_aux,attn_aux_all, out_aux_all, out_aux_2_all, pred_domain_aux_all, feat_aux_all = model(aux_data, aux, beta, mu, is_train=True, reverse=False)
        
        # ignore dummy tensors
        # Update mem bank
        feat_aux_pos, feat_aux_neg = feat_aux_all[1][:feat_aux[1].shape[0]], feat_aux_all[1][feat_aux[1].shape[0]:]
        attn_source, out_source, out_source_2, pred_domain_source, feat_source = removeDummy(attn_source, out_source, out_source_2, pred_domain_source, feat_source, batch_source_ori)
        attn_target, out_target, out_target_2, pred_domain_target, feat_target = removeDummy(attn_target, out_target, out_target_2, pred_domain_target, feat_target, batch_target_ori)
        attn_source_pos, out_source_pos, out_source_2_pos, pred_domain_source_pos, feat_source_pos = removeDummy(attn_source_pos, out_source_pos, out_source_2_pos, pred_domain_source_pos, feat_source_pos, batch_source_ori)
        attn_target_pos, out_target_pos, out_target_2_pos, pred_domain_target_pos, feat_target_pos = removeDummy(attn_target_pos, out_target_pos, out_target_2_pos, pred_domain_target_pos, feat_target_pos, batch_target_ori)
        attn_source_neg, out_source_neg, out_source_2_neg, pred_domain_source_neg, feat_source_neg = removeDummy(attn_source_neg, out_source_neg, out_source_2_neg, pred_domain_source_neg, feat_source_neg, batch_source_ori)
        attn_target_neg, out_target_neg, out_target_2_neg, pred_domain_target_neg, feat_target_neg = removeDummy(attn_target_neg, out_target_neg, out_target_2_neg, pred_domain_target_neg, feat_target_neg, batch_target_ori)
        if args.pred_normalize == 'Y': # use the uncertainly method (in contruction...)
            out_source = out_source / out_source.var().log()
            out_target = out_target / out_target.var().log()
        '''for label in range(num_class):
            mem_bank[label].extend([item for item in feat_source[1][source_label==label]])
            #mem_bank[label].extend([item for item in feat_target[1][target_label==label]])
            if len(mem_bank[label]) > 4000:
                mem_bank[label] = mem_bank[label][len(mem_bank[label])-4000:]
        loss_nce = 0.0
        if epoch > 2:
            for label in range(num_class):
                #print(prototypes.shape)
                prototype = prototypes[label]
                anchor_samples = feat_source[1][(source_label==label).cuda()] # bs, 256
                prototype = prototype.unsqueeze(0).repeat(anchor_samples.shape[0],1) # bs, 256
                mem_bank_samples = torch.stack(mem_bank[label]).cuda() # 4000, 256
                #print(mem_bank_samples.shape)
                #print(anchor_samples.shape)
                l_neg = torch.einsum('nc,ck->nk', [anchor_samples, mem_bank_samples.transpose(1,0)])
                k = 5
                if l_neg.shape[0] < 5:
                    k = l_neg.shape[0]
                _, topkdix = torch.topk(l_neg.sum(-1), k, dim=0)
                negative_anchors = []
                for index in topkdix:
                    negative_anchors.append(mem_bank_samples[index])
                if len(negative_anchors) == 0:
                    continue
                negative_anchors = torch.stack(negative_anchors,0)
                loss_nce += kld(anchor_samples, prototype.detach(), torch.ones(anchor_samples.shape[0]).cuda()) #info_nce_loss(anchor_samples, prototype.detach(), negative_anchors.detach())
            loss_nce = loss_nce / 80'''
        # store the embedding
        if args.tensorboard:
            feat_source_display = feat_source[1] if i==0 else torch.cat((feat_source_display, feat_source[1]), 0)
            label_source_display = label_source if i==0 else torch.cat((label_source_display, label_source), 0)
            label_source_domain_display = torch.zeros(label_source.size(0)) if i==0 else torch.cat((label_source_domain_display, torch.zeros(label_source.size(0))), 0)
            feat_target_display = feat_target[1] if i==0 else torch.cat((feat_target_display, feat_target[1]), 0)
            label_target_display = label_target if i==0 else torch.cat((label_target_display, label_target), 0)
            label_target_domain_display = torch.ones(label_target.size(0)) if i==0 else torch.cat((label_target_domain_display, torch.ones(label_target.size(0))), 0)
        out = out_source
        label = label_source

        out = torch.cat((out, out_target))
        label = torch.cat((label, label_target))
        loss_classification = criterion(out, label) + 0.01*criterion(out_aux, aux_label)
        if args.ens_DA == 'MCD' and args.use_target != 'none':
            loss_classification += criterion(out_source_2, label)
        losses_c.update(loss_classification.item(), out_source.size(0))
        center_list = [[] for k in range(num_class)]
        centers = torch.zeros(num_class, 256).cuda()
        count = torch.zeros(num_class).cuda()
        for feat, l in zip(feat_target[1], target_label):
            center_list[l].append(feat)
        for k in range(num_class):
            if len(center_list[k]) != 0:
                centers[k] = torch.stack(center_list[k],0).mean(0)
        selected_centers = []
        mask = torch.isnan(centers)
        #print(mask)
        centers[mask] = 0.0*centers[mask]
        for feat, l in zip(feat_source[1], source_label):
            if torch.sum(centers[l.long()]) !=0:
                selected_centers.append(centers[l.long()])
            else:
                selected_centers.append(feat)
        selected_centers = torch.stack(selected_centers,0)
        #print(selected_centers)
        loss = loss_classification + 0.1*info_nce_loss(feat_source[1], selected_centers, torch.cat([feat_target_neg[1], feat_source_neg[1]],0)) + 0.1*info_nce_loss(feat_aux[1], feat_aux_pos, feat_aux_neg)#+ 0.1*info_nce_loss(feat_target[1], feat_target_pos[1], feat_target_neg[1])
        #+ 0.01*info_nce_loss(feat_source[1], feat_source_pos[1], feat_target_neg[1]) + 0.01*info_nce_loss(feat_target[1], feat_target_pos[1], feat_source_neg[1])
        '''source_domain_label = torch.ones(out_source.shape[0]).cuda()
        target_domain_label = torch.zeros(out_target.shape[0]).cuda()
        domain_label = torch.cat((source_domain_label, target_domain_label), 0)
        domain_label = domain_label.cuda(non_blocking=True)
        pred_domain = torch.cat([pred_domain_source[0], pred_domain_target[0]],0)
        loss_dompred =torch.nn.functional.mse_loss(pred_domain.squeeze(), domain_label.float())
        loss += 1 * loss_dompred.float()''' 
        # measure accuracy and record loss
        pred = out
        #print(label.shape)
        prec1, prec5 = accuracy(pred.contiguous(), label, topk=(1, 5))

        losses.update(loss.item())
        top1.update(prec1.item(), out_source.size(0))
        top5.update(prec5.item(), out_source.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient and args.verbose:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            line = 'Train: [{0}][{1}/{2}], lr: {lr:.5f}\t' + \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' + \
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' + \
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t' + \
                    'Loss {loss.val:.4f} ({loss.avg:.4f})   loss_c {loss_c.avg:.4f}\t'

            if args.dis_DA != 'none' and args.use_target != 'none':
                line += 'alpha {alpha:.3f}  loss_d {loss_d.avg:.4f}\t'

            if args.adv_DA != 'none' and args.use_target != 'none':
                line += 'beta {beta[0]:.3f}, {beta[1]:.3f}, {beta[2]:.3f}  loss_a {loss_a.avg:.4f}\t'

            if args.add_loss_DA != 'none' and args.use_target != 'none':
                line += 'gamma {gamma:.6f}  loss_e {loss_e.avg:.4f}\t'

            if args.ens_DA != 'none' and args.use_target != 'none':
                line += 'mu {mu:.6f}  loss_s {loss_s.avg:.4f}\t'

            line = line.format(
                epoch, i, len(source_loader), batch_time=batch_time, data_time=data_time, alpha=alpha, beta=beta, gamma=gamma, mu=mu,
                loss=losses, loss_c=losses_c, loss_d=losses_d, loss_a=losses_a, loss_e=losses_e, loss_s=losses_s, top1=top1, top5=top5,
                lr=optimizer.param_groups[0]['lr'])

            if i % args.show_freq == 0:
                print(line)

            log.write('%s\n' % line)

        # adjust the learning rate for ech step (e.g. DANN)
        if args.lr_adaptive == 'dann':
            adjust_learning_rate_dann(optimizer, p)

        # save attention values w/ the selected class
        if args.save_attention >= 0:
            attn_source = attn_source[source_label==args.save_attention]
            attn_target = attn_target[target_label==args.save_attention]
            attn_epoch_source = torch.cat((attn_epoch_source, attn_source.cpu()))
            attn_epoch_target = torch.cat((attn_epoch_target, attn_target.cpu()))

    # update the embedding every epoch
    if args.tensorboard:
        n_iter_train = epoch * len(source_loader) # calculate the total iteration
        # embedding
        # see source and target separately
        writer.add_embedding(feat_source_display, metadata=label_source_display.data, global_step=n_iter_train, tag='train_source')
        writer.add_embedding(feat_target_display, metadata=label_target_display.data, global_step=n_iter_train, tag='train_target')

        # mix source and target
        feat_all_display = torch.cat((feat_source_display, feat_target_display), 0)
        label_all_domain_display = torch.cat((label_source_domain_display, label_target_domain_display), 0)
        writer.add_embedding(feat_all_display, metadata=label_all_domain_display.data, global_step=n_iter_train, tag='train_DA')

        # emphazise some classes (1, 3, 11 here)
        label_source_1 = 1 * torch.eq(label_source_display, torch.cuda.LongTensor([1]).repeat(label_source_display.size(0))).long().cuda(non_blocking=True)
        label_source_3 = 2 * torch.eq(label_source_display, torch.cuda.LongTensor([3]).repeat(label_source_display.size(0))).long().cuda(non_blocking=True)
        label_source_11 = 3 * torch.eq(label_source_display, torch.cuda.LongTensor([11]).repeat(label_source_display.size(0))).long().cuda(non_blocking=True)

        label_target_1 = 4 * torch.eq(label_target_display, torch.cuda.LongTensor([1]).repeat(label_target_display.size(0))).long().cuda(non_blocking=True)
        label_target_3 = 5 * torch.eq(label_target_display, torch.cuda.LongTensor([3]).repeat(label_target_display.size(0))).long().cuda(non_blocking=True)
        label_target_11 = 6 * torch.eq(label_target_display, torch.cuda.LongTensor([11]).repeat(label_target_display.size(0))).long().cuda(non_blocking=True)

        label_source_display_new = label_source_1 + label_source_3 + label_source_11
        id_source_show = ~torch.eq(label_source_display_new, 0).cuda(non_blocking=True)
        label_source_display_new = label_source_display_new[id_source_show]
        feat_source_display_new = feat_source_display[id_source_show]

        label_target_display_new = label_target_1 + label_target_3 + label_target_11
        id_target_show = ~torch.eq(label_target_display_new, 0).cuda(non_blocking=True)
        label_target_display_new = label_target_display_new[id_target_show]
        feat_target_display_new = feat_target_display[id_target_show]

        feat_all_display_new = torch.cat((feat_source_display_new, feat_target_display_new), 0)
        label_all_display_new = torch.cat((label_source_display_new, label_target_display_new), 0)
        writer.add_embedding(feat_all_display_new, metadata=label_all_display_new.data, global_step=n_iter_train, tag='train_DA_labels')

    log_short.write('%s\n' % line)
    return losses_c.avg, attn_epoch_source.mean(0), attn_epoch_target.mean(0)


def calculate_target_cluster_center(target_loader, model):
	Target_centers_list = [[] for i in range(12)]
	Target_centers = torch.zeros(12,256)
	counter = torch.zeros(12)
	with torch.no_grad():
		for i, (target_data, target_label) in enumerate(target_loader):
			target_size_ori = target_data.size()  # original shape
			batch_target_ori = target_size_ori[0]
			# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
			if batch_target_ori < args.batch_size[0]:
				target_data_dummy = torch.zeros(args.batch_size[0] - batch_target_ori, target_size_ori[1], target_size_ori[2])
				target_data = torch.cat((target_data, target_data_dummy))
			# add dummy tensors to make sure batch size can be divided by gpu #
			if target_data.size(0) % gpu_count != 0:
				target_data_dummy = torch.zeros(gpu_count - target_data.size(0) % gpu_count, target_data.size(1), target_data.size(2))
				target_data = torch.cat((target_data, target_data_dummy))
			target_label = target_label.cuda(non_blocking=True) # pytorch 0.4.X
			target_data = target_data.cuda()
			if args.baseline_type == 'frame':
				target_label_frame = target_label.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames
			label_target = target_label_frame if args.baseline_type == 'frame' else target_label  # determine the label for calculating the loss function
			attn_source, out_source, out_source_2, pred_domain_source, feat_source, attn_target, out_target, out_target_2, pred_domain_target, feat_target = model(target_data, target_data, [0]*len(args.beta), 0, is_train=False, reverse=False)

			attn_source, out_source, out_source_2, pred_domain_source, feat_source = removeDummy(attn_source, out_source, out_source_2, pred_domain_source, feat_source, batch_target_ori)
			attn_target, out_target, out_target_2, pred_domain_target, feat_target = removeDummy(attn_target, out_target, out_target_2, pred_domain_target, feat_target, batch_target_ori)
			for k in range(feat_target[1].shape[0]):
				Target_centers[target_label[k].cpu()] += feat_target[1][k].cpu()/torch.norm(feat_target[1][k], p='fro', dim=-1, keepdim=False, out=None, dtype=None).detach().cpu()
				counter[target_label[k]] += 1
			del attn_source, out_source, out_source_2, pred_domain_source, feat_source, attn_target, out_target, out_target_2, pred_domain_target, feat_target, target_data, target_label
		target_centers = []
		for i in range(12):
			tensor_list = Target_centers[i]
			target_centers.append(tensor_list.cuda()/counter[i])
		target_centers = torch.stack(target_centers).cuda()
	return target_centers

def validate_plus(source_loader, target_loader, val_loader, model, criterion, num_class, epoch, log):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	top1_source = AverageMeter()
	top5_source = AverageMeter()

	top1_target = AverageMeter()
	top5_target = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()

	# initialize the embedding
	if args.tensorboard:
		feat_val_display = None
		label_val_display = None
	'''Source_centers_list = [[] for i in range(8)]
	Source_centers = [[] for i in range(8)]
	for i, (source_data, source_label) in enumerate(source_loader):

		source_size_ori = source_data.size()  # original shape
		batch_source_ori = source_size_ori[0]
		# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
		if batch_source_ori < args.batch_size[0]:
			source_data_dummy = torch.zeros(args.batch_size[0] - batch_source_ori, source_size_ori[1], source_size_ori[2])
			source_data = torch.cat((source_data, source_data_dummy))
		# add dummy tensors to make sure batch size can be divided by gpu #
		if source_data.size(0) % gpu_count != 0:
			source_data_dummy = torch.zeros(gpu_count - source_data.size(0) % gpu_count, source_data.size(1), source_data.size(2))
			source_data = torch.cat((source_data, source_data_dummy))
		# measure data loading time
		#data_time.update(time.time() - end)
		source_label = source_label.cuda(non_blocking=True) # pytorch 0.4.X
		source_data = source_data.cuda()
		if args.baseline_type == 'frame':
			source_label_frame = source_label.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames
		label_source = source_label_frame if args.baseline_type == 'frame' else source_label  # determine the label for calculating the loss function
		attn_source, out_source, out_source_2, pred_domain_source, feat_source, attn_target, out_target, out_target_2, pred_domain_target, feat_target = model(source_data, source_data, [0]*len(args.beta), 0, is_train=False, reverse=False)
		for k in range(source_data.shape[0]):
			Source_centers[source_label[k]].append(feat_source[1][k]/torch.norm(feat_source[1][k], p='fro', dim=None, keepdim=False, out=None, dtype=None))
	source_centers = []
	for i in range(8):
		tensor_list = torch.stack(Source_centers[i])
		source_centers.append(tensor_list.mean(0))
	source_centers = torch.stack(source_centers)'''



	target_centers = calculate_target_cluster_center(source_loader, model)
	for i, (val_data, val_label) in enumerate(val_loader):

		val_size_ori = val_data.size()  # original shape
		batch_val_ori = val_size_ori[0]

		# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
		if batch_val_ori < args.batch_size[2]:
			val_data_dummy = torch.zeros(args.batch_size[2] - batch_val_ori, val_size_ori[1], val_size_ori[2])
			val_data = torch.cat((val_data, val_data_dummy))

		# add dummy tensors to make sure batch size can be divided by gpu #
		if val_data.size(0) % gpu_count != 0:
			val_data_dummy = torch.zeros(gpu_count - val_data.size(0) % gpu_count, val_data.size(1), val_data.size(2))
			val_data = torch.cat((val_data, val_data_dummy))

		val_label = val_label.cuda(non_blocking=True)
		with torch.no_grad():

			if args.baseline_type == 'frame':
				val_label_frame = val_label.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames

			# compute output
			#print(val_data.shape)
			_, _, _, _, _, attn_val, out_val, out_val_2, pred_domain_val, feat_val = model(val_data, val_data, [0]*len(args.beta), 0, is_train=False, reverse=False)

			# ignore dummy tensors
			attn_val, out_val, out_val_2, pred_domain_val, feat_val = removeDummy(attn_val, out_val, out_val_2, pred_domain_val, feat_val, batch_val_ori)

			# measure accuracy and record loss
			label = val_label_frame if args.baseline_type == 'frame' else val_label

			# store the embedding
			if args.tensorboard:
				feat_val_display = feat_val[1] if i == 0 else torch.cat((feat_val_display, feat_val[1]), 0)
				label_val_display = label if i == 0 else torch.cat((label_val_display, label), 0)

			pred = out_val

			if args.baseline_type == 'tsn':
				pred = pred.view(val_label.size(0), -1, num_class).mean(dim=1) # average all the segments (needed when num_segments != val_segments)

			loss = criterion(pred, label)

			preds_source = []
			preds_target = []
			for feat in feat_val[1]:
				feat = feat / torch.norm(feat, p='fro', dim=-1, keepdim=False, out=None, dtype=None)
				dist_source = torch.sqrt((torch.abs(feat.unsqueeze(0).repeat(12,1) - target_centers)**2).mean(-1))
				dist_target = torch.sqrt((torch.abs(feat.unsqueeze(0).repeat(12,1) - target_centers)**2).mean(-1))
				preds_source.append(1-dist_source)#(torch.argmin(dist_source, dim=0))
				preds_target.append(1-dist_target)#(torch.argmin(dist_target, dim=0))
			preds_source = torch.stack(preds_source)
			preds_target = torch.stack(preds_target)
			
			prec1, prec5 = accuracy(pred.data, label, topk=(1, 5))
			prec1_source, prec5_source = accuracy(preds_source.data, label, topk=(1, 5))
			prec1_target, prec5_target = accuracy(preds_target.data, label, topk=(1, 5))

			losses.update(loss.item(), out_val.size(0))
			top1.update(prec1.item(), out_val.size(0))
			top5.update(prec5.item(), out_val.size(0))
			top1_source.update(prec1_source.item(), out_val.size(0))
			top5_source.update(prec5_source.item(), out_val.size(0))
			top1_target.update(prec1_target.item(), out_val.size(0))
			top5_target.update(prec5_target.item(), out_val.size(0))
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				line = 'Test: [{0}][{1}/{2}]\t' + \
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' + \
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' + \
					  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'+ \
					  'Prec@1_source {top1_source.val:.3f} ({top1_source.avg:.3f})\t' + \
					  'Prec@5_source {top5_source.val:.3f} ({top5_source.avg:.3f})\t'+ \
					  'Prec@1_target {top1_target.val:.3f} ({top1_target.avg:.3f})\t' + \
					  'Prec@5_target {top5_target.val:.3f} ({top5_target.avg:.3f})\t'
				line = line.format(
					   epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
					   top1=top1, top5=top5, top1_source=top1_source, top1_target=top1_target, top5_source=top5_source, top5_target=top5_target)

				if i % args.show_freq == 0:
					print(line)

				log.write('%s\n' % line)

	if args.tensorboard:  # update the embedding every iteration
		# embedding
		n_iter_val = epoch * len(val_loader)

		writer.add_embedding(feat_val_display, metadata=label_val_display.data, global_step=n_iter_val, tag='validation')

	print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Prec@1_source {top1_source.avg:.3f} Prec@5_source {top5_source.avg:.3f}  Prec@1_target {top1_target.avg:.3f} Prec@5_target {top5_target.avg:.3f}  Loss {loss.avg:.5f}'
		  .format(top1=top1, top5=top5, top1_source=top1_source, top5_source=top5_source,top1_target=top1_target, top5_target=top5_target, loss=losses)))

	return top1.avg




def validate(val_loader, model, criterion, num_class, epoch, log):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()

	# initialize the embedding
	if args.tensorboard:
		feat_val_display = None
		label_val_display = None

	for i, (val_data, val_label,_,_) in enumerate(val_loader):

		val_size_ori = val_data.size()  # original shape
		batch_val_ori = val_size_ori[0]

		# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
		if batch_val_ori < args.batch_size[2]:
			val_data_dummy = torch.zeros(args.batch_size[2] - batch_val_ori, val_size_ori[1], val_size_ori[2])
			val_data = torch.cat((val_data, val_data_dummy))

		# add dummy tensors to make sure batch size can be divided by gpu #
		if val_data.size(0) % gpu_count != 0:
			val_data_dummy = torch.zeros(gpu_count - val_data.size(0) % gpu_count, val_data.size(1), val_data.size(2))
			val_data = torch.cat((val_data, val_data_dummy))

		val_label = val_label.cuda(non_blocking=True)
		with torch.no_grad():

			if args.baseline_type == 'frame':
				val_label_frame = val_label.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames

			# compute output
			#print(val_data.shape)
			_, _, _, _, _, attn_val, out_val, out_val_2, pred_domain_val, feat_val = model(val_data, val_data, [0]*len(args.beta), 0, is_train=False, reverse=False)

			# ignore dummy tensors
			attn_val, out_val, out_val_2, pred_domain_val, feat_val = removeDummy(attn_val, out_val, out_val_2, pred_domain_val, feat_val, batch_val_ori)

			# measure accuracy and record loss
			label = val_label_frame if args.baseline_type == 'frame' else val_label

			# store the embedding
			if args.tensorboard:
				feat_val_display = feat_val[1] if i == 0 else torch.cat((feat_val_display, feat_val[1]), 0)
				label_val_display = label if i == 0 else torch.cat((label_val_display, label), 0)

			pred = out_val

			if args.baseline_type == 'tsn':
				pred = pred.view(val_label.size(0), -1, num_class).mean(dim=1) # average all the segments (needed when num_segments != val_segments)

			loss = criterion(pred, label)
			prec1, prec5 = accuracy(pred.data, label, topk=(1, 5))

			losses.update(loss.item(), out_val.size(0))
			top1.update(prec1.item(), out_val.size(0))
			top5.update(prec5.item(), out_val.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				line = 'Test: [{0}][{1}/{2}]\t' + \
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' + \
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' + \
					  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'

				line = line.format(
					   epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
					   top1=top1, top5=top5)

				if i % args.show_freq == 0:
					print(line)

				log.write('%s\n' % line)

	if args.tensorboard:  # update the embedding every iteration
		# embedding
		n_iter_val = epoch * len(val_loader)

		writer.add_embedding(feat_val_display, metadata=label_val_display.data, global_step=n_iter_val, tag='validation')

	print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
		  .format(top1=top1, top5=top5, loss=losses)))

	return top1.avg


def save_checkpoint(state, is_best, path_exp, filename='checkpoint.pth.tar'):

	path_file = path_exp + filename
	torch.save(state, path_file)
	if is_best:
		path_best = path_exp + 'model_best.pth.tar'
		shutil.copyfile(path_file, path_best)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, decay):
	"""Sets the learning rate to the initial LR decayed by 10 """
	for param_group in optimizer.param_groups:
		param_group['lr'] /= decay

def adjust_learning_rate_loss(optimizer, decay, stat_current, stat_previous, op):
	ops = {'>': (lambda x, y: x > y), '<': (lambda x, y: x < y), '>=': (lambda x, y: x >= y), '<=': (lambda x, y: x <= y)}
	if ops[op](stat_current, stat_previous):
		for param_group in optimizer.param_groups:
			param_group['lr'] /= decay

def adjust_learning_rate_dann(optimizer, p):
	for param_group in optimizer.param_groups:
		param_group['lr'] = args.lr / (1. + 10 * p) ** 0.75

def loss_adaptive_weight(loss, pred):
	weight = 1 / pred.var().log()
	constant = pred.std().log()
	return loss * weight + constant

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].contiguous().view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

# remove dummy tensors
def removeDummy(attn, out_1, out_2, pred_domain, feat, batch_size):
	attn = attn[:batch_size]
	out_1 = out_1[:batch_size]
	out_2 = out_2[:batch_size]
	pred_domain = [pred[:batch_size] for pred in pred_domain]
	feat = [f[:batch_size] for f in feat]

	return attn, out_1, out_2, pred_domain, feat

if __name__ == '__main__':
	main()
