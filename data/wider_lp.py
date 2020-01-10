import torch
import torch.utils.data as data
import cv2
import numpy as np


class WiderLPDetection(data.Dataset):

    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path, 'r')
        lines = f.readlines()
        
        for line in lines:
            line = line.strip('\n').split(',')
            self.imgs_path.append(line[0])
            # lable = list(map(int, line[3:]))
            lable = list(map(int, line[1:]))
            self.words.append([lable]) 
        
    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])        
        labels = self.words[index]

        annotations = np.zeros((0, 13))
        if len(labels) == 0:
            return annotations
        
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 13))
            '''
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[2]  # x2
            annotation[0, 3] = label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]  # l0_x
            annotation[0, 5] = label[5]  # l0_y
            annotation[0, 6] = label[6]  # l1_x
            annotation[0, 7] = label[7]  # l1_y
           
            annotation[0, 8] = label[8]  # l3_x
            annotation[0, 9] = label[9]  # l3_y
            annotation[0, 10] = label[10]  # l4_x
            annotation[0, 11] = label[11]  # l4_y
            if (annotation[0, 4] < 0):
                annotation[0, 12] = -1
            else:
                annotation[0, 12] = 1
            ''' 
            # bbox
            annotation[0, 0] = label[2]  # x1
            annotation[0, 1] = label[3]  # y1
            annotation[0, 2] = label[4]  # x2
            annotation[0, 3] = label[5]  # y2

            # landmarks
            annotation[0, 4] = label[6]  # l0_x
            annotation[0, 5] = label[7]  # l0_y
            annotation[0, 6] = label[8]  # l1_x
            annotation[0, 7] = label[9]  # l1_y
           
            annotation[0, 8] = label[10]  # l3_x
            annotation[0, 9] = label[11]  # l3_y
            annotation[0, 10] = label[12]  # l4_x
            annotation[0, 11] = label[13]  # l4_y
            if (label[0] < 0):
                annotation[0, 12] = -1
            else:
                annotation[0, 12] = 1 
                        
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations).astype(np.float64)
        
        if self.preproc is not None:

            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
