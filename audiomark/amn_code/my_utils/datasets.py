import logging

import torch
from pathlib import Path

import webrtcvad

import audiomark.amn_code.my_utils.utils as utils
import torchaudio
import numpy as np
import os
from torchaudio.datasets import SPEECHCOMMANDS
from operator import itemgetter
from torch.utils.data import Dataset
import pickle
from PIL import Image
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, EMNIST, GTSRB, FashionMNIST
from tqdm import tqdm


class MySpeechCmd(SPEECHCOMMANDS):
    def __init__(self, root, subset, transform, frequency_transform, meta_dir,
                 specify_labels=None, specify_name="", include=True, expected_sec=1.0):
        super().__init__(root=root, download=True)

        self.transform = transform
        self.unique_labels = None
        self.expected_sec = expected_sec
        self.expected_sr = 16000

        self.frequency_transform = frequency_transform
        # self.frequency_transform = torchaudio.transforms.MFCC(sample_rate=self.expected_sr, melkwargs={"n_fft": 1024})

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
        else:
            assert False, f"invalid subset: {subset}"

        assert specify_labels is not None, "must specify labels for now"

        # load saved index or get required indexes
        if not meta_dir.exists():
            Path.mkdir(meta_dir, parents=True)

        specify_idxes_path = meta_dir.joinpath(f"MySpeechCmd_{subset}_{specify_labels}_include_{include}_{specify_name}_idxes.bin")
        if specify_idxes_path.exists():
            with open(specify_idxes_path, 'rb') as handle:
                specify_idxes = pickle.load(handle)
        else:
            specify_idxes = []
            for i in tqdm(range(len(self))):
                label = self[i][1]
                if (include is True) and (label in specify_labels):
                    # specify_labels are for inclusion
                    specify_idxes.append(i)
                elif (include is False) and (label not in specify_labels):
                    # specify_labels are for exclusion
                    specify_idxes.append(i)

            with open(specify_idxes_path, 'wb') as handle:
                pickle.dump(specify_idxes, handle)

        self._walker = list(itemgetter(*specify_idxes)(self._walker))

        # load all labels if exists
        all_labels_path = meta_dir.joinpath(f"MySpeechCmd_{subset}_{specify_labels}_include_{include}_{specify_name}_all_labels.bin")
        if all_labels_path.exists():
            with open(all_labels_path, 'rb') as handle:
                all_labels = pickle.load(handle)
        else:
            all_labels = [x[1] for x in self]
            with open(all_labels_path, 'wb') as handle:
                pickle.dump(all_labels, handle)

        self.unique_labels = list(set(all_labels))
        self.unique_labels.sort()       # from now on, getitem will return label index instead of string

        # load targets if exists
        targets_path = meta_dir.joinpath(f"MySpeechCmd_{subset}_{specify_labels}_include_{include}_{specify_name}_targets.bin")
        if targets_path.exists():
            with open(targets_path, 'rb') as handle:
                self.targets = pickle.load(handle)
        else:
            self.targets = [x[1] for x in self]
            with open(targets_path, 'wb') as handle:
                pickle.dump(self.targets, handle)

        self.targets = torch.LongTensor(self.targets)

        if specify_labels is not None:
            # check validity
            if include is True:
                assert len(self.unique_labels) == len(specify_labels)
                for x in self.unique_labels:
                    assert x in specify_labels
            else:
                for x in self.unique_labels:
                    assert x not in specify_labels

    def __getitem__(self, item):
        item_data = super().__getitem__(item)
        item_data = list(item_data)
        item_data[0] = item_data[0].squeeze()       # eliminate the extra dim

        waveform, sr, label, *_ = item_data

        # we expect the length of input is expected_sec
        expect_size = int(sr * self.expected_sec)
        wav_len = waveform.shape[0]
        if wav_len < expect_size:
            tmp = torch.zeros(expect_size)
            tmp[: wav_len] = waveform
            waveform = tmp
        waveform = waveform[: expect_size]

        if self.frequency_transform is not None:
            data = self.frequency_transform(waveform.squeeze())
        else:
            data = waveform.squeeze()

        data = data.unsqueeze(0)
        # apply transform and model on whole batch directly on device
        if self.transform is not None:
            data = self.transform(data)

        # return index instead of string if self.unique_labels is set
        if self.unique_labels is not None:
            label = self.unique_labels.index(label)

        return data.cpu(), label


class CelebA(Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, id_path, selected_attrs, transform, selected_ids):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = Path(image_dir)
        self.attr_path = attr_path
        self.id_path = id_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.selected_ids = selected_ids

        self.img_path_lst = []
        self.targets = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        self.num_images = len(self.img_path_lst)
        self.targets = torch.LongTensor(self.targets).squeeze()

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        lines_id = [line.rstrip() for line in open(self.id_path, 'r')]
        all_attr_names = lines[1].split()  # This is 1 because line 0 is the len
        all_attr_names.append('id')

        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        for i, line in enumerate(lines):
            split_id = lines_id[i].split()
            split = line.split()

            assert split[0] == split_id[0], "file names must coincide."

            filename = split[0]
            values = split[1:]
            id = split_id[1]
            values.append(id)

            if (self.selected_ids is not None) and (int(id) not in self.selected_ids):
                # skip if this id is not selected
                continue

            label = []
            for j, attr_name in enumerate(self.selected_attrs):
                idx = self.attr2idx[attr_name]
                # for the id, we do separate
                if attr_name == 'id':
                    label.append(int(values[idx]))
                else:
                    label.append(values[idx] == '1')

            self.img_path_lst.append(filename)
            self.targets.append(label)

        print(f'Finished preprocessing the CelebA dataset. '
              f'There are {len(self.selected_ids)} ids and {len(self.img_path_lst)} images...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename, label = self.img_path_lst[index], self.targets[index]
        # print(filename, label)
        image = Image.open(self.image_dir.joinpath(filename))
        return self.transform(image), label

    def __len__(self):
        """Return the number of images."""
        return self.num_images


# EMNIST data need to transpose pixels
class MyEMNIST(EMNIST):
    def __init__(self, root: str, split: str, **kwargs):
        super().__init__(root, split, **kwargs)

        # let the targets starting from 0 instead of 1
        self.targets = self.targets - 1

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        x = torch.transpose(x, 1, 2)

        return x, y


class MyGTSRB(GTSRB):
    def __init__(self, root, split, transform=None, target_transform=None, download=True,
                 specify_labels=None, meta_dir=None, min_avg_val=0.0):
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform,
                         download=download)

        self.specify_labels = specify_labels
        if self.specify_labels is None:
            self.specify_labels = list(range(43))  # 43 classes in total

        meta_path = None
        if meta_dir is not None:
            # read saved meta if exists
            meta_path = meta_dir.joinpath(f"{split}_MinAvg_{min_avg_val:.1f}_labels_{self.specify_labels}.bin")

            if meta_path.exists():
                with open(meta_path, 'rb') as handle:
                    (self.idx_lst, self.targets) = pickle.load(handle)
                self.targets = torch.LongTensor(self.targets)
                return

        self.idx_lst = []
        self.targets = []

        logging.info("iterating GTSRB...")
        total_num = super().__len__()
        for idx in tqdm(range(total_num)):
            x, y = super().__getitem__(idx)

            # set min_avg_val to a higher value to skip images that are too dark
            if y in self.specify_labels and x.mean() >= min_avg_val:
                self.idx_lst.append(idx)
                self.targets.append(self.specify_labels.index(y))

        if meta_path is not None:
            if not meta_dir.exists():
                Path.mkdir(meta_dir)
            with open(meta_path, 'wb') as handle:
                pickle.dump((self.idx_lst, self.targets), handle)

        self.targets = torch.LongTensor(self.targets)

    def __len__(self):
        return len(self.idx_lst)

    def __getitem__(self, idx):
        idx = self.idx_lst[idx]
        x, y = super().__getitem__(idx)
        y = self.specify_labels.index(y)

        return torch.clip(x, 0, 1), y


class MyFashionMNIST(Dataset):
    def __init__(self, root, train, transform, target_transform=None, download=False,
                 specify_labels=None, meta_dir=None, normalize=False):
        super().__init__()

        self.fmnist = FashionMNIST(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.normalize = normalize

        self.specify_labels = specify_labels
        if self.specify_labels is None:
            self.specify_labels = list(range(len(self.fmnist.classes)))        # total number of classes
            self.idx_lst = range(len(self.fmnist))
            self.targets = self.fmnist.targets
            return

        meta_path = None
        if meta_dir is not None:
            # read saved meta if exists
            meta_path = meta_dir.joinpath(f"train_{train}_labels_{self.specify_labels}.bin")

            if meta_path.exists():
                with open(meta_path, 'rb') as handle:
                    (self.idx_lst, self.targets) = pickle.load(handle)
                self.targets = torch.LongTensor(self.targets)
                return

        self.idx_lst = []
        self.targets = []

        logging.info(f"iterating {self.__class__.__name__}...")
        total_num = len(self.fmnist)
        for idx in tqdm(range(total_num)):
            x, y = self.fmnist[idx]

            if y in self.specify_labels:
                self.idx_lst.append(idx)
                self.targets.append(self.specify_labels.index(y))

        if meta_path is not None:
            if not meta_dir.exists():
                Path.mkdir(meta_dir)
            with open(meta_path, 'wb') as handle:
                pickle.dump((self.idx_lst, self.targets), handle)

        self.targets = torch.LongTensor(self.targets)

    def __len__(self):
        return len(self.idx_lst)

    def __getitem__(self, idx):

        y = self.targets[idx].item()
        org_idx = self.idx_lst[idx]
        x, org_y = self.fmnist[org_idx]

        if self.normalize:
            # normalize x into [0, 1]
            assert x.max() > x.min(), "cannot divided by 0"
            x = (x - x.min()) / (x.max() - x.min())

        assert y == self.specify_labels.index(org_y)
        assert x.min().item() >= 0 and x.max().item() <= 1

        return x, y


class MNISTSubset(MNIST):
    def __init__(
            self,
            root: str,
            idx_start_perc=None,
            idx_end_perc=None,
            train: bool = True,
            transform=None,
            target_transform=None,
            download: bool = False,
    ) -> None:
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

        total_len = super().__len__()

        # [idx_start,  idx_end): idx_end is excluded
        self.idx_start = 0 if idx_start_perc is None else int(idx_start_perc * total_len)
        self.idx_end = super().__len__() if idx_end_perc is None else int(idx_end_perc * total_len)

        assert self.__len__() <= total_len, "This must be a subset."

    def __len__(self):
        return self.idx_end - self.idx_start

    def __getitem__(self, index):
        dest_idx = index + self.idx_start
        assert dest_idx < self.idx_end

        return super().__getitem__(dest_idx)



class MyCIFAR100(CIFAR100):
    def __init__(
            self,
            root,
            train,
            transform=None,
            target_transform=None,
            download=False,

    ) -> None:
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

        # self.targets return torch.Tensor instead of list to be the same as MNIST
        self.targets = torch.LongTensor(self.targets)


class MiniImgageNet(Dataset):
    def __init__(self, data_dir, preprocess, train, dset_path):
        super().__init__()
        self.preprocess = preprocess
        dset_path = Path(dset_path)

        self.data = None
        self.targets = None
        if dset_path is not None:
            # load existing data if possible
            if dset_path.exists():
                loaded_data = np.load(dset_path)
                self.data, self.targets = loaded_data["data"], loaded_data["labels"]
                assert self.data.shape[0] == self.targets.shape[0]

                self.data = torch.FloatTensor(self.data)
                self.targets = torch.LongTensor(self.targets)

        if self.data is None:
            # no existing data, we load original data, convert them and save
            self._load_all_data(data_dir, train)
            if not dset_path.parent.exists():
                Path.mkdir(dset_path.parent)

            np.savez(dset_path, data=self.data.numpy(), labels=self.targets.numpy())

    def _load_all_data(self, data_dir, train):
        path_lst = list(data_dir.glob("*.jpg"))
        path_lst.sort()

        num = len(path_lst)
        train_num = int(num * 0.9)
        valid_idxes = np.random.RandomState(seed=21531).permutation(num)
        if train is True:
            valid_idxes = valid_idxes[:train_num]
        else:
            valid_idxes = valid_idxes[train_num:]

        img_path_list = []
        for idx in tqdm(valid_idxes):
            img_path_list.append(path_lst[idx])

        # we do not care the labels, so just return a value
        self.targets = torch.arange(0, len(img_path_list)) % 10

        # load all data into memory
        self.data = []
        logging.info("loading all images into memory...")
        for path in tqdm(img_path_list):
            self.data.append(self._load_img(path))

        self.data = torch.stack(self.data)

    def _load_img(self, path):
        img = Image.open(path)
        img = self.preprocess(img)
        return img

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class MyCIFAR10(CIFAR10):
    def __init__(
            self,
            root,
            train,
            transform=None,
            target_transform=None,
            download=False,
    ) -> None:
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

        # self.targets return torch.Tensor instead of list to be the same as MNIST
        self.targets = torch.LongTensor(self.targets)


class MySubset(torch.utils.data.Dataset):
    def __init__(self, dset, rand_subset_split=None, subset_flip=False, rand_subset_seed=1292):
        super().__init__()
        self.dset = dset
        self.rand_subset_split = rand_subset_split

        total_len = len(self.dset)

        if rand_subset_split is None:
            self.idxes = list(range(total_len))  # full dataset
        else:
            self.idxes = np.random.RandomState(seed=rand_subset_seed).permutation(total_len)
            subset_len = int(total_len * rand_subset_split)
            if subset_flip is False:
                self.idxes = self.idxes[: subset_len]
            else:
                # use the remaining data as the subset
                self.idxes = self.idxes[subset_len:]
            logging.info(f"{type(self.dset).__name__} subset length = {len(self.idxes)}: {self.idxes[:10]} ...")

        self.targets = self.dset.targets

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, index):
        dest_idx = self.idxes[index]
        return self.dset[dest_idx]


class MyASVSpoof2019(torch.utils.data.Dataset):

    # index from the protocol files
    IDX_SPEAKER_ID = 0
    IDX_AUDIO_FILE_NAME = 1
    IDX_SYSTEM_ID = 3
    IDX_KEY = 4

    DATASET_NAME = "ASVspoof2019"
    SR = 16000  # ASVSpoof2019 sampling rate is 16000
    AGGRESSIVENESS = 3

    def __init__(self, root, track, subset, transform=None, frequency_transform=None,
                 noise_reduction=True, autogreg=True, fft_n=512, fft_remove=64, extra_num=float("inf"),
                 ret_full_info=True):
        super().__init__()

        self.transform = transform
        self.frequency_transform = frequency_transform

        self.noise_reduction = noise_reduction
        self.vad = webrtcvad.Vad(self.AGGRESSIVENESS)

        self.autogreg = autogreg
        self.fft_n = fft_n
        self.fft_remove = fft_remove
        self.extra_num = extra_num

        self.ret_full_info = ret_full_info

        assert subset in ["train", "dev", "eval"], "the subset must be 'train', 'dev', or 'eval'!"
        if subset == "train":
            pro_suffix = "trn.txt"
        else:
            pro_suffix = "trl.txt"

        root = Path(root)
        track_dir = root.joinpath(f"{self.DATASET_NAME}/{track}")
        pro_path = track_dir.joinpath(f"{self.DATASET_NAME}_{track}_cm_protocols/"
                                      f"{self.DATASET_NAME}.{track}.cm.{subset}.{pro_suffix}")

        # read in all the protocols
        self.protos = []
        with open(pro_path, "r") as f:
            all_lines = f.readlines()
        for line in all_lines:
            line = line.strip().split(" ")

            assert line[self.IDX_KEY] in ["bonafide", "spoof"]

            self.protos.append(
                utils.ConfigBase(
                    speaker_id=line[self.IDX_SPEAKER_ID],
                    fname=line[self.IDX_AUDIO_FILE_NAME] + ".flac",
                    system_id=line[self.IDX_SYSTEM_ID],
                    key=line[self.IDX_KEY]
                )
            )

        # data dir that stores all the audios
        self.flac_dir = track_dir.joinpath(f"{self.DATASET_NAME}_{track}_{subset}/flac")

    def __len__(self):
        return len(self.protos)

    def extract_feq_seq(self, wav):
        """
        extract sequences in the frequency domain for autoregressive training
        """
        freq_abs = utils.stft_to_abs(wav, wav.device, n_fft=self.fft_n)
        freq_abs = freq_abs[self.fft_remove:, :]

        pre_len = 5
        freq_len = freq_abs.shape[1]
        extra_num = self.extra_num

        batch_x = []
        batch_y = []

        rand_pos = np.random.permutation(freq_len - pre_len - 1) + 1        # [1, freq_len - pre_len)
        total_batch_size = min(rand_pos.shape[0], extra_num)

        for batch_idx in range(total_batch_size):
            i = rand_pos[batch_idx]
            batch_x.append(freq_abs[:, i: i + pre_len].unsqueeze(0))

            # bidirectional autoregressive
            y = [freq_abs[:, i + pre_len], freq_abs[:, i - 1]]
            y = torch.concat(y)
            batch_y.append(y.unsqueeze(0))

        batch_x = torch.concat(batch_x, dim=0)
        batch_y = torch.concat(batch_y, dim=0)

        return batch_x, batch_y

    def __getitem__(self, item):
        item_data = self.protos[item]

        # load wav file
        wav, sr = torchaudio.backend.sox_io_backend.load(filepath=self.flac_dir.joinpath(item_data.fname))
        assert sr == self.SR
        wav = wav.squeeze()       # eliminate the extra dim
        assert len(wav.shape) == 1, "only one channel allowed"

        label = 0 if item_data.key == "bonafide" else 1

        if self.frequency_transform is not None:
            wav = self.frequency_transform(wav)

        # apply transform and model on whole batch directly on device
        if self.transform is not None:
            wav = wav.unsqueeze(0)
            wav = self.transform(wav).squeeze()

        if self.noise_reduction is True:
            org_wav = wav
            wav = utils.remove_non_audio(wav.numpy(), self.vad, sr=self.SR)
            if wav.shape[0] <= 0.1 * self.SR:
                wav = org_wav
            else:
                wav = torch.FloatTensor(wav)

        if self.autogreg is True:
            freq_x, freq_y = self.extract_feq_seq(wav)
            rep_times = freq_x.shape[0]
            label = [label] * rep_times

            if self.ret_full_info is True:
                return freq_x, freq_y, label, [item_data.system_id] * rep_times
            return freq_x, freq_y, label

        return wav, label, item_data.system_id


def MyASVSpoof2019_freq_seq_collate(batch):

    x = [item[0] for item in batch]
    x = torch.concat(x, dim=0)

    y = [item[1] for item in batch]
    y = torch.concat(y, dim=0)

    labels = [item[2] for item in batch]

    def flatten_lst(_lst):
        flat_list = []
        for sublist in _lst:
            flat_list.extend(sublist)
        return flat_list

    labels = torch.LongTensor(flatten_lst(labels))

    if len(batch[0]) == 3:
        return [x, (y, labels)]

    elif len(batch[0]) == 4:
        sys_ids = [item[3] for item in batch]
        sys_ids = flatten_lst(sys_ids)
        return x, y, labels, sys_ids

    else:
        assert False, "something wrong with the dataset???"


def MyASVSpoof2019_wav_collate(batch):
    assert len(batch[0]) == 3, "something wrong with the dataset???"

    x = [item[0] for item in batch]
    x = torch.concat(x, dim=0)

    y = [item[1] for item in batch]
    y = torch.LongTensor(y)

    sys_ids = [item[2] for item in batch]

    return x, y, sys_ids


class MyNumpyData(Dataset):
    def __init__(self, np_data, np_labels, preprocess=None):
        super().__init__()

        self.np_data = np_data
        self.np_labels = np_labels

        self.preprocess = preprocess
        self.targets = torch.LongTensor(self.np_labels)     # to be compatible with other datasets

    def __len__(self):
        return self.np_data.shape[0]

    def __getitem__(self, idx):

        x, y = torch.FloatTensor(self.np_data[idx]), self.np_labels[idx]

        if self.preprocess is not None:
            x = self.preprocess(x)

        return x, y

    def add_extra_data(self, extra_data, extra_labels):

        if self.np_data is None:
            self.np_data = extra_data
            self.np_labels = extra_labels
            return

        self.np_data = np.concatenate([self.np_data, extra_data], axis=0)
        self.np_labels = np.concatenate([self.np_labels, extra_labels])


class MyNumpyAudioData(Dataset):
    def __init__(self, np_data_lst, np_wm_lst, expected_x_len, preprocess=None, normalize_if_less=None):
        super().__init__()

        self.np_data_lst = np_data_lst
        self.np_wm_lst = np_wm_lst
        self.expected_x_len = expected_x_len
        self.preprocess = preprocess

        self.targets = torch.LongTensor([0] * len(np_data_lst))     # to be compatible with other datasets

        if normalize_if_less is not None:
            assert 0 <= normalize_if_less <= 1.0
            # adjust the range of values
            for idx, data in enumerate(self.np_data_lst):
                if np.abs(data).max() >= normalize_if_less:
                    continue

                min_val, max_val = data.min(), data.max()
                data = (data - min_val) / (max_val - min_val)       # to [0, 1]
                data = data * 2 - 1                                 # to [-1, 1]
                data = data * normalize_if_less           # to [-normalize_if_less, normalize_if_less]
                self.np_data_lst[idx] = data

    def __len__(self):
        return len(self.np_data_lst)

    def __getitem__(self, idx):

        x = np.zeros(self.expected_x_len)

        org_x = self.np_data_lst[idx]
        if len(org_x) <= len(x):
            x[:len(org_x)] = org_x
        else:
            # randomly get one section
            rand_pos = np.random.randint(len(org_x) - len(x) + 1)
            sec = org_x[rand_pos: rand_pos + len(x)]
            assert len(sec) == len(x)
            x = sec

        x = torch.FloatTensor(x)

        if self.preprocess is not None:
            x = self.preprocess(x)

        wm = torch.from_numpy(self.np_wm_lst[idx])

        return (x, wm), 0

    def add_extra_data(self, extra_data, extra_em):

        if self.np_data_lst is None:
            self.np_data_lst = extra_data
            self.np_wm_lst = extra_em
            return

        self.np_data_lst = np.concatenate([self.np_data_lst, extra_data], axis=0)
        self.np_wm_lst = np.concatenate([self.np_wm_lst, extra_em])


class PairedNumpyAudioData(Dataset):
    def __init__(self, src_data_lst, tgt_data_lst, expected_x_len, preprocess=None):
        super().__init__()

        self.src_data_lst = src_data_lst
        self.tgt_data_lst = tgt_data_lst
        assert len(src_data_lst) == len(tgt_data_lst), "something wrong with the dataset???"

        self.expected_x_len = expected_x_len
        self.preprocess = preprocess

        self.targets = torch.LongTensor([0] * len(src_data_lst))     # to be compatible with other datasets

    def __len__(self):
        return len(self.src_data_lst)

    def __getitem__(self, idx):
        assert self.src_data_lst[idx].shape[0] == self.tgt_data_lst[idx].shape[0], "length should be same"

        def rand_slice(data, _rand_pos=None):
            x = np.zeros(self.expected_x_len)
            if len(data) <= len(x):
                x[:len(data)] = data
            else:
                # randomly get one section
                if _rand_pos is None:
                    _rand_pos = np.random.randint(len(data) - len(x) + 1)
                sec = data[_rand_pos: _rand_pos + len(x)]
                assert len(sec) == len(x)
                x = sec

            x = torch.FloatTensor(x)
            return x, _rand_pos

        src_slice, rand_pos = rand_slice(self.src_data_lst[idx])
        tgt_slice, _ = rand_slice(self.tgt_data_lst[idx], rand_pos)

        if self.preprocess is not None:
            src_slice = self.preprocess(src_slice)
            tgt_slice = self.preprocess(tgt_slice)

        return src_slice, tgt_slice
















