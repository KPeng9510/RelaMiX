# RelaMiX: Exploring Few-Shot Adaptation inVideo-based Action Recognition

---
## Contents

## Requirements
* support Python 3.6, PyTorch 0.4, CUDA 9.0, CUDNN 7.1.4
* install all the library with: `pip install -r requirements.txt`
## Dataset Preparation
### Data structure
You need to extract frame-level features for each video to run the codes. To extract features, please check [`dataset_preparation/`](dataset_preparation/).

Folder Structure:
```
DATA_PATH/
  DATASET/
    list_DATASET_SUFFIX.txt
    RGB/
      CLASS_01/
        VIDEO_0001.mp4
        VIDEO_0002.mp4
        ...
      CLASS_02/
      ...

    RGB-Feature/
      VIDEO_0001/
        img_00001.t7
        img_00002.t7
        ...
      VIDEO_0002/
      ...
```
`RGB-Feature/` contains all the feature vectors for training/testing. `RGB/` contains all the raw videos.

There should be at least two `DATASET` folders: source training set  and validation set. If you want to do domain adaption, you need to have another `DATASET`: target training set.

### File lists for training/validation


### Input data


