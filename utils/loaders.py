import glob
import math
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from abc import ABC
import pandas as pd
from .epic_record import EpicVideoRecord
from .actionnet_record import ActionNetRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from utils.logger import logger

''' 
CLASSES USED FOR THIS SETTING
        Verbs:
        0 - take (get)
        1 - put-down (put/place)
        2 - open
        3 - close
        4 - wash (clean)
        5 - cut
        6 - stir (mix)
        7 - pour
        Domains:
        D1 - P08
        D2 - P01
        D3 - P22
'''

class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features",
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def _get_train_indices(self, record, modality='RGB'):
        if self.dense_sampling[modality]:
            # selecting one frame and discarding another (alternation), to avoid duplicates
            center_frames = np.linspace(0, record.num_frames[modality], self.num_clips + 2,
                                        dtype=np.int32)[1:-1]

            indices = [x for center in center_frames for x in
                       range(center - math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride),
                             # start of the segment
                             center + math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride),
                             # end of the segment
                             self.stride)]  # step of the sampling

            offset = -indices[0] if indices[0] < 0 else 0
            for i in range(0, len(indices), self.num_frames_per_clip[modality]):
                indices_old = indices[i]
                for j in range(self.num_frames_per_clip[modality]):
                    indices[i + j] = indices[i + j] + offset if indices_old < 0 else indices[i + j]

            return indices

        else:
            indices = []
            # average_duration is the average stride among frames in the clip to obtain a uniform sampling BUT
            # the randint shifts a little (to add randomicity among clips)
            average_duration = record.num_frames[modality] // self.num_frames_per_clip[modality]
            if average_duration > 0:
                for _ in range(self.num_clips):
                    frame_idx = np.multiply(list(range(self.num_frames_per_clip[modality])), average_duration) + \
                                randint(average_duration, size=self.num_frames_per_clip[modality])
                    indices.extend(frame_idx.tolist())
            else:
                indices = np.zeros((self.num_frames_per_clip[modality] * self.num_clips,))
            indices = np.asarray(indices)

        return indices

    def _get_val_indices(self, record, modality):
        max_frame_idx = max(1, record.num_frames[modality])
        if self.dense_sampling[modality]:
            n_clips = self.num_clips
            center_frames = np.linspace(0, record.num_frames[modality], n_clips + 2, dtype=np.int32)[1:-1]

            indices = [x for center in center_frames for x in
                       range(center - math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride),
                             # start of the segment
                             center + math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride),
                             # end of the segment
                             self.stride)]  # step of the sampling

            offset = -indices[0] if indices[0] < 0 else 0
            for i in range(0, len(indices), self.num_frames_per_clip[modality]):
                indices_old = indices[i]
                for j in range(self.num_frames_per_clip[modality]):
                    indices[i + j] = indices[i + j] + offset if indices_old < 0 else indices[i + j]

            return indices

        else:  # uniform sampling
            # Code for "Deep Analysis of CNN-based Spatio-temporal Representations for Action Recognition"
            # arXiv: 2104.09952v1
            # Yuan Zhi, Zhan Tong, Limin Wang, Gangshan Wu
            frame_idices = []
            sample_offsets = list(range(-self.num_clips // 2 + 1, self.num_clips // 2 + 1))
            for sample_offset in sample_offsets:
                if max_frame_idx > self.num_frames_per_clip[modality]:
                    tick = max_frame_idx / float(self.num_frames_per_clip[modality])
                    curr_sample_offset = sample_offset
                    if curr_sample_offset >= tick / 2.0:
                        curr_sample_offset = tick / 2.0 - 1e-4
                    elif curr_sample_offset < -tick / 2.0:
                        curr_sample_offset = -tick / 2.0
                    frame_idx = np.array([int(tick / 2.0 + curr_sample_offset + tick * x) for x
                                          in range(self.num_frames_per_clip[modality])])
                else:
                    np.random.seed(sample_offset - (-self.num_clips // 2 + 1))
                    frame_idx = np.random.choice(max_frame_idx, self.num_frames_per_clip[modality])
                frame_idx = np.sort(frame_idx)
                frame_idices.extend(frame_idx.tolist())
            frame_idx = np.asarray(frame_idices)
            return frame_idx

    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]

        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            if len(sample_row) != 1:
                print(f"Exited because len is {len(sample_row)}, with uid: {int(record.uid)}")
                exit(-1)
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl
        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added
            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        elif modality == 'Flow':
            idx_untrimmed = (record.start_frame // 2) + idx
            try:
                x_img = Image.open(os.path.join(data_path, record.untrimmed_video_name,
                                                tmpl.format('x', idx_untrimmed))).convert('L')
                y_img = Image.open(os.path.join(data_path, record.untrimmed_video_name,
                                                tmpl.format('y', idx_untrimmed))).convert('L')
            except FileNotFoundError:
                for i in range(0, 3):
                    found = True
                    try:
                        x_img = Image.open(os.path.join(data_path, record.untrimmed_video_name,
                                                        tmpl.format('x', idx_untrimmed + i))).convert('L')
                        y_img = Image.open(os.path.join(data_path, record.untrimmed_video_name,
                                                        tmpl.format('y', idx_untrimmed + i))).convert('L')
                    except FileNotFoundError:
                        found = False

                    if found:
                        break
            return [x_img, y_img]

        elif modality == 'Event':
            idx_untrimmed = (record.start_frame // self.dataset_conf["Event"].rgb4e) + idx
            try:
                img_npy = np.load(os.path.join(data_path, record.untrimmed_video_name,
                                               tmpl.format(idx_untrimmed))).astype(np.float32)
            except FileNotFoundError:
                max_idx_video = int(sorted(glob.glob(os.path.join(self.dataset_conf['RGB'].data_path,
                                                                  record.untrimmed_video_name, "img_*")))[-1]
                                    .split("_")[-1].split(".")[0])
                if max_idx_video % 6 == 0:
                    max_idx_event = (max_idx_video // self.dataset_conf["Event"].rgb4e) - 1
                else:
                    max_idx_event = max_idx_video // self.dataset_conf["Event"].rgb4e
                if idx_untrimmed > max_idx_video:
                    img_npy = np.load(os.path.join(data_path, record.untrimmed_video_name,
                                                   tmpl.format(max_idx_event))).astype(np.float32)
                else:
                    raise FileNotFoundError
            return np.stack([img_npy], axis=0)
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)

class ActionNetDataset(data.Dataset, ABC):

    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs) -> None:
        """
        - split: ActionNet if modality is EMG or S04 if RGB
        - modalities can be RGB(not implemented yet) and EMG data
        - mode is a string (train, test)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        super().__init__()

        self.modalities = modalities  # considered modalities (ex. [RGB,EMG])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        else:
            pickle_name = split + "_test.pkl"
        
        
        self.list_file = pd.read_pickle(os.path.join(dataset_conf.annotations_path, pickle_name))
        #print(f'list_val_load: {self.list_file}, add: {os.path.join(self.dataset_conf.annotations_path, pickle_name)}')
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [ ActionNetRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat
        
        

        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                if kwargs:
                    #logger.info(f'jeez : saved_features/{self.dataset_conf[m].features_name}_{pickle_name}')
                    model_features = pd.DataFrame(pd.read_pickle(os.path.join(self.dataset_conf[m].features_name + "_" + pickle_name))['features'])[["uid", "features_" + m]]
                else:
                    #logger.info(f'jeez : saved_features/{self.dataset_conf[m].features_name}_{pickle_name}')
                    model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features", self.dataset_conf[m].features_name + "_" + pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")
                self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")
        
    
    def _get_train_indices(self, record, modality='RGB'):
        if self.dense_sampling[modality]:
            # selecting one frame and discarding another (alternation), to avoid duplicates
            center_frames = np.linspace(0, record.num_frames[modality], self.num_clips + 2,
                                        dtype=np.int32)[1:-1]

            indices = [x for center in center_frames for x in
                       range(center - math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride),
                             # start of the segment
                             center + math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride),
                             # end of the segment
                             self.stride)]  # step of the sampling

            offset = -indices[0] if indices[0] < 0 else 0
            for i in range(0, len(indices), self.num_frames_per_clip[modality]):
                indices_old = indices[i]
                for j in range(self.num_frames_per_clip[modality]):
                    indices[i + j] = indices[i + j] + offset if indices_old < 0 else indices[i + j]
            return indices

        else:
            indices = []
            # average_duration is the average stride among frames in the clip to obtain a uniform sampling BUT
            # the randint shifts a little (to add randomicity among clips)
            average_duration = record.num_frames[modality] // self.num_frames_per_clip[modality]
            if average_duration > 0:
                for _ in range(self.num_clips):
                    frame_idx = np.multiply(list(range(self.num_frames_per_clip[modality])), average_duration) + \
                                randint(average_duration, size=self.num_frames_per_clip[modality])
                    indices.extend(frame_idx.tolist())
            else:
                indices = np.zeros((self.num_frames_per_clip[modality] * self.num_clips,))
            indices = np.asarray(indices)

        return indices

    def _get_val_indices(self, record, modality):
        max_frame_idx = max(1, record.num_frames[modality])
        if self.dense_sampling[modality]:
            n_clips = self.num_clips
            center_frames = np.linspace(0, record.num_frames[modality], n_clips + 2, dtype=np.int32)[1:-1]

            indices = [x for center in center_frames for x in
                       range(center - math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride),
                             # start of the segment
                             center + math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride),
                             # end of the segment
                             self.stride)]  # step of the sampling

            offset = -indices[0] if indices[0] < 0 else 0
            for i in range(0, len(indices), self.num_frames_per_clip[modality]):
                indices_old = indices[i]
                for j in range(self.num_frames_per_clip[modality]):
                    indices[i + j] = indices[i + j] + offset if indices_old < 0 else indices[i + j]

            return indices

        else:  
            # uniform sampling
            # Code for "Deep Analysis of CNN-based Spatio-temporal Representations for Action Recognition"
            # arXiv: 2104.09952v1
            # Yuan Zhi, Zhan Tong, Limin Wang, Gangshan Wu
            frame_idices = []
            sample_offsets = list(range(-self.num_clips // 2 + 1, self.num_clips // 2 + 1))
            for sample_offset in sample_offsets:
                if max_frame_idx > self.num_frames_per_clip[modality]:
                    tick = max_frame_idx / float(self.num_frames_per_clip[modality])
                    curr_sample_offset = sample_offset
                    if curr_sample_offset >= tick / 2.0:
                        curr_sample_offset = tick / 2.0 - 1e-4
                    elif curr_sample_offset < -tick / 2.0:
                        curr_sample_offset = -tick / 2.0
                    frame_idx = np.array([int(tick / 2.0 + curr_sample_offset + tick * x) for x
                                          in range(self.num_frames_per_clip[modality])])
                else:
                    np.random.seed(sample_offset - (-self.num_clips // 2 + 1))
                    frame_idx = np.random.choice(max_frame_idx, self.num_frames_per_clip[modality])
                frame_idx = np.sort(frame_idx)
                frame_idices.extend(frame_idx.tolist())
            frame_idx = np.asarray(frame_idices)
            return frame_idx

    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]
       
        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = torch.stack([torch.Tensor(sample_row["features_" + m].values[i]) for i in range(len(sample_row))])
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        if modality == 'EMG':
            # n_fft control the number of frequency bin bin=n_fft // 2+1
            n_fft = 2*(self.num_frames_per_clip[modality] - 1)
            win_length = None
            hop_length = 1
            # print(f'nfft +{n_fft}')
            spectrogram = T.Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=False,
                pad_mode="reflect",
                power=2.0,
                normalized=True
            )
            
            # legge lo spectrogramma di tutto il video, ha dimensione 160*durata del video(in s)
            readings = {
                'left': record.myo_left_readings,
                'right': record.myo_right_readings
            }

            # print(f" [ DEBUG ] - right: {len(readings['left'])} samples, left: {len(readings['right'])} samples")
            freq = {}
            result = []
            if indices[-1] > len(readings['left']):
                print("NON SI PUòòòò")
                exit(9)
            for arm in ['left', 'right']:
                signal = torch.from_numpy(readings[arm]).float()
                #print(signal)
                freq[arm] = [spectrogram(signal[:, i]) for i in range(8)]
                for channel in freq[arm]:
                    # print(f" [ DEBUG ] - {arm} in freq has {channel.shape} samples")
                    # print(f"[ DEBUG ] indices: {len(indices)}, from {indices[0]} to {indices[-1]}")
                    # print(f"[ DEBUG ] spec_indices: {len(indices)}, from {indices[0]} to {indices[-1]}")
                    result.append(torch.stack([channel[:, i] for i in indices]))
            result = torch.stack(result)

            if self.transform is None:
                return result, record.label
            process_data = self.transform[modality](result)
            return process_data, record.label

        else:
            images = list()
            for frame_index in indices:
                p = int(frame_index)
                # here the frame is loaded in memory
                frame = self._load_data(modality, record, p)
                images.extend(frame)
            # finally, all the transformations are applied
            if self.transform is None:
                return images, record.label
            process_data = self.transform[modality](images)
            return process_data, record.label


    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl
        if modality == 'RGB':
            # here the offset for the starting index of the sample is added
            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, tmpl.format(idx_untrimmed))).convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,"img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path,tmpl.format(max_idx_video))).convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)