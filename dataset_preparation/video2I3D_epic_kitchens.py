import os
import imageio.v2 as imageio
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image
import torchvision.transforms as transforms
import torch
import time
import pickle
import cv2
#from scipy.misc import imresize
import random
cv2.setNumThreads(0)
import time
import sys
import glob
from lavis.models import load_model_and_preprocess

def loadVideo(filepath, rescale=None, verbose=False, start_frame = 0, n_frames = 0):
    """
    Extracts all frames of a video and combines them to a ndarray with
    shape (frame id, height, width, channels)
    Parameters
    ----------
    filepath: str
        path to video file including the video name
        (e.g '/your/file/video.avi')
    rescale: str
        rescale input video to desired resolution (e.g. rescale='160x120')
    verbose: bool
        hide or display debug information
    Returns
    -------
    Numpy: frames
        all frames of the video with (frame id, height, width, channels)
    """
    cap = cv2.VideoCapture(filepath)

    if (start_frame > 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame);

    # Interate over frames in video
    images = []
    count = 0
    max_video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT) - start_frame-1
    
    #print(start_frame)
    if (n_frames > max_video_length):
        #print(filepath, start_frame, n_frames)
        #print ("n_frames >= max_video_length")
        pass

    if (n_frames > 0 and n_frames <= max_video_length):
        video_length = n_frames
    else:
        video_length = max_video_length
    while cap.isOpened():
        #print('test')
        ret, image = cap.read()
        #print(image)
        if rescale:
            #Convert to PIL image for rescaling
            image = Image.fromarray(image)
            image = image.resize((224,224), resample=Image.BILINEAR)
            # Convert back to numpy array
            image = np.array(image)
        count = count + 1
        if (count > video_length -1 or (len(np.shape(image))<2)):

            cap.release()
            # Print stats
            if (verbose):
                print("Done extracting frames.\n%d frames extracted" % count)
                print("-----")
                print(np.shape(image))
                print(np.shape(images))
            break

        images.append(image)
    if verbose:
        print(np.shape(images))

        print(filepath)
        print(np.shape(images))
        print(start_frame)
        print(n_frames)
        print("---")

    return images

batch_size = 8

def im2tensor(im):
    im = Image.fromarray(im)
    t_im = data_transform(im)
    return t_im

def extract_frame_feature_batch(list_tensor):
    with torch.no_grad():
        batch_tensor = torch.stack(list_tensor)
        batch_tensor = torch.transpose(batch_tensor, 0, 1)

        batch_tensor = batch_tensor.unsqueeze(0) 
        features = extractor(batch_tensor)

        features = features.view(features.size(0), -1).cpu()    
        return features


from pytorch_i3d import InceptionI3d
model = InceptionI3d(400, in_channels=3)
model.load_state_dict(torch.load('/hkfs/work/workspace_haic/scratch/fy2374-hiwi_workspace/ZDDA/pytorch-i3d/models/rgb_imagenet.pt'))
extractor = torch.nn.DataParallel(model.cuda())
extractor.eval()

path_input = '/hkfs/work/workspace_haic/scratch/fy2374-acmmm/epic_kitchen/'
path_output = '//hkfs/work/workspace_haic/scratch/fy2374-hiwi_workspace/epic_kitchen/features/'
csv_file = '/hkfs/work/workspace_haic/scratch/fy2374-hiwi_workspace/ZDDA/MM-SADA-code/Annotations/D2_train.pkl'
list_path = '../dataset/epic-kitchens/list/'
with open(csv_file, 'rb') as f:
    dataset_pd = pickle.load(f)
uid = dataset_pd["uid"].to_numpy()
start_frame = dataset_pd["start_frame"].to_numpy()
stop_frame = dataset_pd["stop_frame"].to_numpy()
video_id = dataset_pd["video_id"].to_numpy()
verb_class = dataset_pd["verb_class"].to_numpy()
if csv_file[:-4].endswith("train"):
    #path_output += "train/"
    #path_input += "train/"
    mode = 'train'
else: 
    #path_output += "test/"
    #path_input += "test/"
    mode = 'test'
stripped_csv_file = csv_file.split("/")[-1]
if stripped_csv_file.startswith("D1"):
    path_output += "P08/"
    path_input += "P08/videos/"
    domain = 'P08'
if stripped_csv_file.startswith("D2"):
    path_output += "P01/"
    path_input += "P01/videos/"
    domain = 'P01'
if stripped_csv_file.startswith("D3"):
    path_output += "P22/"
    path_input += "P22/videos/"
    domain = 'P22'

feature_in_type = '.t7'

data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

start = time.time()
path_list = []
label_list = []
save_list = []
for i in range(len(video_id)):
    path_list.append(path_input + video_id[i] + '.MP4')
    save_list.append(path_output + video_id[i] + '/')
    label_list.append(verb_class[i]) 
#mode = 'train'
list_file = list_path + 'list_%s_%s.txt' % (domain, mode)
file1 = open(list_file, "w")

for j in range(len(path_list)):
    print(j)
    if j != 2380:
        continue
    if j > 2380:
        break
    if path_list[j].split('/')[-1].split('.')[0] != 'P01_18':
        continue
    if path_list[j].split('/')[-1].split('.')[0] == 'P22_10':
        rgb_files = [i for i in os.listdir(path_list[j].replace('.MP4', '').replace('videos', 'rgb_frames').replace('P22_10', '')) if i.endswith('.jpg')]
        rgb_files.sort()
        frames_tensor = []
        save_path = save_list[j] + str(j)
        numframes = len(glob.glob(save_path+"/img_*****.t7", recursive = True))
        #print(numframes)
        if not os.path.isdir(save_list[j] + str(j)):
            os.makedirs(save_list[j] + str(j))
        if (numframes) >= (stop_frame[j]-start_frame[j]-1):
            print(path_list[j], 'already done')
            continue
        #print(path_list[j])
        repath = path_list[j].replace('.MP4', '').replace('videos', 'rgb_frames').replace('P22_10', '')
        print(start_frame[i])
        #try:
        list_frames = loadVideo(path_list[j],  start_frame=start_frame[j], n_frames=(stop_frame[j]-start_frame[j]))#rgb_files[start_frame[j]:stop_frame[j]]
        list_frames = rgb_files[start_frame[j]:stop_frame[j]]

        list_content = save_list[j][3:] + str(j) + ' ' + str(len(list_frames)) + ' ' + str(label_list[j]) + '\n'
        #file1.write(list_content)

        #num_files = len([i for i in os.listdir(save_path)])

        try:
            for t in range(len(list_frames)):
                im = imageio.imread(repath + '/' + list_frames[t])
                if np.sum(im.shape) != 0:
                    id_frame = t+1
                    frames_tensor.append(im2tensor(im))
        except RuntimeError:
            print('Could not read frame', id_frame+1, 'from', video_file)
        
        num_frames = len(frames_tensor)
        
        num_numpy = int(batch_size/2) 
        for i in range(num_numpy):
            frames_tensor = [torch.zeros_like(frames_tensor[0])] + frames_tensor
            frames_tensor.append(torch.zeros_like(frames_tensor[0]))

        features = torch.Tensor()

        for t in range(0, num_frames, 1):
            frames_batch = frames_tensor[t:t+batch_size]    
            features_batch = extract_frame_feature_batch(frames_batch)
            features = torch.cat((features,features_batch))

        for t in range(features.size(0)):
            id_frame = t+1
            id_frame_name = str(id_frame).zfill(5)
            filename =save_path + '/img_' + id_frame_name + feature_in_type            

            if not os.path.exists(filename):
                torch.save(features[t].clone(), filename)
        print('Save ' + save_path + ' done!')
    else:
        frames_tensor = []
        save_path = save_list[j] + str(j)
        numframes = len(glob.glob(save_path+"/img_*****.t7", recursive = True))
        print(numframes)
        if not os.path.isdir(save_list[j] + str(j)):
            os.makedirs(save_list[j] + str(j))
        if (numframes) >= (stop_frame[j]-start_frame[j]-1):   
            print(path_list[j], 'already done')
            continue
        
        list_frames = loadVideo(path_list[j], start_frame=start_frame[j], n_frames=(stop_frame[j]-start_frame[j]))
        list_content = save_list[j][3:] + str(j) + ' ' + str(len(list_frames)) + ' ' + str(label_list[j]) + '\n'
        #file1.write(list_content)
        num_files = len([i for i in os.listdir(save_path)])

        try:
            for t in range(len(list_frames)):
                im = list_frames[t]#imageio.imread(path_list[j] + '/' + list_frames[t])
                if np.sum(im.shape) != 0:
                    id_frame = t+1
                    frames_tensor.append(im2tensor(im))
        except RuntimeError:
            print('Could not read frame', id_frame+1, 'from', video_file)

        num_frames = len(frames_tensor)

        num_numpy = int(batch_size/2) 
        for i in range(num_numpy):
            frames_tensor = [torch.zeros_like(frames_tensor[0])] + frames_tensor
            frames_tensor.append(torch.zeros_like(frames_tensor[0]))

        features = torch.Tensor()

        for t in range(0, num_frames, 1):
            frames_batch = frames_tensor[t:t+batch_size]    
            features_batch = extract_frame_feature_batch(frames_batch)
            features = torch.cat((features,features_batch))

        for t in range(features.size(0)):
            id_frame = t+1
            id_frame_name = str(id_frame).zfill(5)
            filename =save_path + '/img_' + id_frame_name + feature_in_type            

            if not os.path.exists(filename):
                torch.save(features[t].clone(), filename)
        print('Save ' + save_path + ' done!')        


end = time.time()
print('Total elapsed time: ' + str(end-start))