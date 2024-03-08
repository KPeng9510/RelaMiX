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

from dataset import *
from models import VideoModel
from loss import *
from opts import parser
from utils.utils import randSelectBatch
import math
from torch.utils.data import WeightedRandomSampler
from colorama import init
from colorama import Fore, Back, Style
import numpy as np
from tensorboardX import SummaryWriter
from info_nce import InfoNCE
from pytorch_metric_learning import miners, losses
from sklearn.mixture import GaussianMixture
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
#torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mem_bank = [[] for i in range(12)]
init(autoreset=True)

best_prec1 = 0
gpu_count = torch.cuda.device_count()

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
    

    #=== Data loading ===#
    print(Fore.CYAN + 'loading data......')

    if args.use_opencv:
        print("use opencv functions")

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
        data_length = 5
    shot=20
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
        loss_c, attn_epoch_source, attn_epoch_target = train(num_class, source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, [], [], alpha, beta, gamma, mu)
        
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



    end_train = time.time()
    print(Fore.CYAN + 'total training time:', end_train - start_train)
    val_best_file.write('%.3f\n' % best_prec1)

    # --- write the total time to log files ---#
    line_time = 'total time: {:.3f} '.format(end_train - start_train)


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
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
 
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
	
		
 
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737. 
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
 
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha
    return calibrated_mean, calibrated_cov



def train(num_class, source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, log, log_short, alpha, beta, gamma, mu):
	source_feat_list = []
	target_feat_list = [[] for i in range(num_class)]
	print(len(target_loader))
	print(len(source_loader))
	for i, (target_data, target_label, target_pos, target_neg) in enumerate(target_loader):
		for data, label in zip(target_data, target_label):
			target_feat_list[label].append(data)
	for i in range(num_class):
		source_feat_list.append([])
	source_label_list = []
	for k, (source_data, source_label, source_pos, source_neg) in enumerate(source_loader):
		for data, label in zip(source_data, source_label):
			source_feat_list[label].append(data)	
	gms = [[GaussianMixture(n_components=1, random_state=0).fit(np.stack(source_feat_list[j])[:,i,:]) for i in range(12)] for j in range(12)]
	#for j in range(num_class):
	#	for i in range(12):
	#		gms[j][i].fit(np.stack(source_feat_list[j],0)[:,i,:])

	means_target = []	
	for i in range(num_class):
		means_target.append(np.mean(np.stack(target_feat_list[i]),0))
	data = []
	for j in range(num_class):
		frames = []
		for i in range(12):
			means = means_target[j][i,:]
			base_means = [gms[k][i].means_[0] for k in range(12)]
			base_convs = [gms[k][i].covariances_[0] for k in range(12)]
			num_sampled = 200
			mean, cov = distribution_calibration(means, base_means, base_convs, k=2)
			sampled_data = np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled)
			frames.append(sampled_data)
		frames = np.stack(frames, 1)
		data.append(frames)
	data = np.concatenate(data,0)
	print(data.shape)
	import pickle as pkl

	f = open('generated_new_data_hmdb_ucf_shot20.pkl', 'wb')
	pkl.dump(file=f, obj=data)
	f.close()



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
