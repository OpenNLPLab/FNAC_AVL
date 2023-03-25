import os
import csv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
import xml.etree.ElementTree as ET
from audio_io import load_audio_av, open_audio_av
import torch

# augment on audio maps
class FrequencyMask(object):
    """
    Implements frequency masking transform from SpecAugment paper (https://arxiv.org/abs/1904.08779)
    
      Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     FrequencyMask(max_width=10, use_mean=False),
        >>> ])
    """

    def __init__(self, max_width, use_mean=True):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) where the frequency mask is to be applied.
        Returns:
            Tensor: Transformed image with Frequency Mask.
        """
        start = random.randrange(0, tensor.shape[2])
        end = start + random.randrange(0, self.max_width)
        if self.use_mean:
            tensor[:, start:end, :] = tensor.mean()
        else:
            tensor[:, start:end, :] = 0
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string


class TimeMask(object):
    """
    Implements time masking transform from SpecAugment paper (https://arxiv.org/abs/1904.08779)
    
      Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     TimeMask(max_width=10, use_mean=False),
        >>> ])
    """

    def __init__(self, max_width, use_mean=True):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) where the time mask is to be applied.
        Returns:
            Tensor: Transformed image with Time Mask.
        """
        start = random.randrange(0, tensor.shape[1])
        end = start + random.randrange(0, self.max_width)
        if self.use_mean:
            tensor[:, :, start:end] = tensor.mean()
        else:
            tensor[:, :, start:end] = 0
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string

def load_image(path):
    return Image.open(path).convert('RGB')


def load_spectrogram(path, dur=3.):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_ss = max(float(audio_dur)/2 - dur/2, 0)
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]

    frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=274)
    spectrogram = np.log(spectrogram + 1e-7)
    # print(spectrogram.shape)
    return spectrogram

def load_all_bboxes(annotation_dir, format='flickr'):
    gt_bboxes = {}
    if format == 'flickr':
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split('.')[0]
            gt = ET.parse(f"{annotation_dir}/{filename}").getroot()
            bboxes = []
            for child in gt:
                for childs in child:
                    bbox = []
                    if childs.tag == 'bbox':
                        for index, ch in enumerate(childs): 
                            if index == 0:
                                continue
                            bbox.append(int(224 * int(ch.text)/256))
                    bboxes.append(bbox)
            gt_bboxes[file] = bboxes

    elif format == 'vggss':
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in annotation['bbox']]
            gt_bboxes[annotation['file']] = bboxes

    return gt_bboxes

def get_all_classes():
    class_list = []
    all_classes={}
    with open('metadata/vggss.json') as json_file:
        annotations = json.load(json_file)
    for annotation in annotations:
        class_name = annotation["class"]
        # all_classes[annotation['file']] = class_name
        if class_name not in class_list:
            class_list.append(class_name)
    
    print('get class amount:', len(class_list))
    for annotation in annotations:
        class_name = annotation["class"]
        all_classes[annotation['file']] = int(class_list.index(class_name))

    return all_classes




def bbox2gtmap(bboxes, format='flickr'):
    gt_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp

    if format == 'flickr':
        # Annotation consensus
        gt_map = gt_map / 2
        gt_map[gt_map > 1] = 1

    elif format == 'vggss':
        # Single annotation
        gt_map[gt_map > 0] = 1

    return gt_map


class AudioVisualDataset(Dataset):
    def __init__(self, image_files, audio_files, image_path, audio_path, audio_dur=3., image_transform=None, audio_transform=None, all_bboxes=None, bbox_format='flickr'):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path
        self.audio_dur = audio_dur

        self.audio_files = audio_files
        self.image_files = image_files
        self.all_bboxes = all_bboxes
        self.bbox_format = bbox_format

        self.image_transform = image_transform
        self.audio_transform = audio_transform

    def getitem(self, idx):
        file = self.image_files[idx]
        file_id = file.split('.')[0]

        # Image
        img_fn = os.path.join(self.image_path, self.image_files[idx])
        frame = self.image_transform(load_image(img_fn))

        # Audio
        audio_fn = os.path.join(self.audio_path, self.audio_files[idx])
        # print(audio_fn)
        spectrogram = self.audio_transform(load_spectrogram(audio_fn))

        bboxes = {}
        if self.all_bboxes is not None:
            bboxes['bboxes'] = self.all_bboxes[file_id]
            bboxes['gt_map'] = bbox2gtmap(self.all_bboxes[file_id], self.bbox_format)

        return frame, spectrogram, bboxes, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])



class AudioVisualDataset2(Dataset):
    def __init__(self, image_files, audio_files, image_path, audio_path, audio_dur=3., image_transform=None, audio_transform=None, all_bboxes=None, bbox_format='flickr', all_classes=None):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path
        self.audio_dur = audio_dur

        self.audio_files = audio_files
        self.image_files = image_files
        self.all_bboxes = all_bboxes
        self.bbox_format = bbox_format

        self.all_classes=all_classes

        self.image_transform = image_transform
        self.audio_transform = audio_transform

    def getitem(self, idx):
        file = self.image_files[idx]
        file_id = file.split('.')[0]

        # Image
        img_fn = os.path.join(self.image_path, self.image_files[idx])
        frame = self.image_transform(load_image(img_fn))

        # Audio
        audio_fn = os.path.join(self.audio_path, self.audio_files[idx])
        spectrogram = self.audio_transform(load_spectrogram(audio_fn))

        bboxes = {}
        if self.all_bboxes is not None:
            bboxes['bboxes'] = self.all_bboxes[file_id]
            bboxes['gt_map'] = bbox2gtmap(self.all_bboxes[file_id], self.bbox_format)
        current_class = None
        if self.all_classes is not None:
            current_class = self.all_classes[file_id]
            current_class = torch.tensor([current_class])
        

        return frame, spectrogram, bboxes, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])

def get_train_dataset(args):
    audio_path = f"{args.train_data_path}/audio/"
    image_path = f"{args.train_data_path}/frames/"

    # audio_path = f"{args.train_data_path}/"
    # image_path = f"{args.train_data_path}/"

    print('trainset:', args.trainset)
    # print(args.trainset.split('_'))
    if args.trainset.split('_')[0] == 'flickr':
        img_format = '.jpg'
    else:
        img_format = '.png'

    print('data path:', audio_path, image_path)
    # print(img_format)

    # List directory
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')}
    image_files = {fn.split(img_format)[0] for fn in os.listdir(image_path) if fn.endswith(img_format)}
    avail_files = audio_files.intersection(image_files)
    print(f"{len(avail_files)} available files")

    # Subsample if specified
    if args.trainset.lower() in {'vggss', 'flickr'}:
        pass    # use full dataset
    else:
        subset = set(open(f"metadata/{args.trainset}.txt").read().splitlines())
        avail_files = avail_files.intersection(subset)
        print(f"{len(avail_files)} valid subset files")
    avail_files = sorted(list(avail_files))
    audio_files = sorted([dt+'.wav' for dt in avail_files])
    image_files = sorted([dt+img_format for dt in avail_files])

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize(int(224 * 1.1), Image.BICUBIC),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        FrequencyMask(max_width=10, use_mean=False),
        TimeMask(max_width=10, use_mean=False),
        transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform
    )

#  for my use, make a little bit change
def get_train_dataset_2(args):
    audio_path = f"{args.train_data_path}/audio/"
    image_path = f"{args.train_data_path}/frames/"

    # audio_path = f"{args.train_data_path}/"
    # image_path = f"{args.train_data_path}/"

    # List directory
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')}
    image_files = {fn.split('.png')[0] for fn in os.listdir(image_path) if fn.endswith('.png')}
    avail_files = audio_files.intersection(image_files)
    print(f"{len(avail_files)} available files")

    # get all classes
    # all_classes = {}
    # if args.trainset == "vggss_10k":
        # all_classes = get_all_classes()

    # Subsample if specified
    if args.trainset.lower() in {'vggss', 'flickr'}:
        pass    # use full dataset
    else:
        subset = set(open(f"metadata/{args.trainset}.txt").read().splitlines())
        avail_files = avail_files.intersection(subset)
        print(f"{len(avail_files)} valid subset files")
    avail_files = sorted(list(avail_files))
    audio_files = sorted([dt+'.wav' for dt in avail_files])
    image_files = sorted([dt+'.png' for dt in avail_files])

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize(int(224 * 1.1), Image.BICUBIC),
        transforms.RandomCrop((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([
        transforms.ToTensor(),
        FrequencyMask(max_width=10, use_mean=False),
        TimeMask(max_width=10, use_mean=False),
        transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        # all_classes=all_classes,
    )


def get_test_dataset(args):
    if args.testset == 'flickr':
        audio_path = args.test_data_path 
        image_path = args.test_data_path 
    elif args.testset == 'vggss':
        audio_path = args.test_data_path + 'audio/'
        image_path = args.test_data_path + 'frames/'

    
    if args.testset == 'flickr':
        testcsv = 'metadata/flickr_test.csv'
    elif args.testset == 'vggss':
        testcsv = 'metadata/vggss_test.csv'
    elif args.testset == 'vggss_heard':
        testcsv = 'metadata/vggss_heard_test.csv'
    elif args.testset == 'vggss_unheard':
        testcsv = 'metadata/vggss_unheard_test.csv'
    else:
        raise NotImplementedError
    bbox_format = {'flickr': 'flickr',
                   'vggss': 'vggss',
                   'vggss_heard': 'vggss',
                   'vggss_unheard': 'vggss'}[args.testset]

    #  Retrieve list of audio and video files
    testset = set([item[0] for item in csv.reader(open(testcsv))])

    # Intersect with available files
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
    avail_files = audio_files.intersection(image_files)
    testset = testset.intersection(avail_files)

    testset = sorted(list(testset))
    image_files = [dt+'.jpg' for dt in testset]
    audio_files = [dt+'.wav' for dt in testset]

    # Bounding boxes
    all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)

    # get all classes
    # all_classes = {}
    # if args.testset == "vggss":
    #     all_classes = get_all_classes()

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        all_bboxes=all_bboxes,
        bbox_format=bbox_format,
    )


def get_test_dataset_2(args):
    audio_path = args.test_data_path + 'audio/'
    image_path = args.test_data_path + 'frames/'

    # audio_path = args.test_data_path 
    # image_path = args.test_data_path 

    if args.testset == 'flickr':
        testcsv = 'metadata/flickr_test.csv'
    elif args.testset == 'vggss':
        testcsv = 'metadata/vggss_test.csv'
    elif args.testset == 'vggss_heard':
        testcsv = 'metadata/vggss_heard_test.csv'
    elif args.testset == 'vggss_unheard':
        testcsv = 'metadata/vggss_unheard_test.csv'
    else:
        raise NotImplementedError
    bbox_format = {'flickr': 'flickr',
                   'vggss': 'vggss',
                   'vggss_heard': 'vggss',
                   'vggss_unheard': 'vggss'}[args.testset]

    #  Retrieve list of audio and video files
    testset = set([item[0] for item in csv.reader(open(testcsv))])

    # Intersect with available files
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
    image_files = {fn.split('.png')[0] for fn in os.listdir(image_path)}
    avail_files = audio_files.intersection(image_files)
    testset = testset.intersection(avail_files)

    testset = sorted(list(testset))
    image_files = [dt+'.png' for dt in testset]
    audio_files = [dt+'.wav' for dt in testset]

    # Bounding boxes
    all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)

    # get all classes
    # all_classes = {}
    # if args.testset == "vggss":
    #     all_classes = get_all_classes()

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        all_bboxes=all_bboxes,
        bbox_format=bbox_format,
        # all_classes=all_classes
    )


def inverse_normalize(tensor):
    inverse_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor



