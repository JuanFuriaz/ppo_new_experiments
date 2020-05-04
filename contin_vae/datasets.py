import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


class RacingcarDataset(Dataset):

    def __init__(self, datadir, train=True, transform=None, target_transform=None, img_stack=1):
        datapath = "%s/train_stack%d.pt" % (datadir, img_stack) if train else "%s/test_stack%d.pt" % (datadir, img_stack)
        self.datadir = datadir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.img_stack = img_stack

        # load data into memory
        self.data, self.targets = torch.load(datapath)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = [self.transform(img[i]) for i in range(img.shape[0])]
            img = torch.stack(img)
            img = img.squeeze(1)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    @staticmethod
    def save_train_test_split(datapath, train_size=0.8, shuffle=True,
                              img_stack=1, cut_frames=0, seed=None):

        # load data from pickle
        data_dict = np.load(datapath, allow_pickle=True)

        # group data into image stacks and reshape to -1 * img_stack * image_shape
        data = []
        for i in range(len(data_dict)):
            episode = data_dict[i]
            # episode should be a list of frames
            episode = [episode[j] for j in range(len(episode))]
            ix_start = len(episode[cut_frames:]) % img_stack
            ix_range = range(ix_start, len(episode[cut_frames:]), img_stack)
            stacks = [[episode[k+l]["state"] for l in range(img_stack)] for k in ix_range]
            data.extend(stacks)
        data = np.array(data, dtype=np.float32)

        # reshape data and targets to -1 * image_shape
        targets = []
        for i in range(len(data_dict)):
            episode = data_dict[i]
            # episode should be a list of frames
            episode = [episode[j] for j in range(len(episode))]
            ix_start = len(episode[cut_frames:]) % img_stack
            ix_range = range(ix_start, len(episode[cut_frames:]), img_stack)
            stacks = [[episode[k+l]["actions"] for l in range(img_stack)] for k in ix_range]
            targets.extend(stacks)
        targets = np.array(targets, dtype=np.float32)

        # # map actions to integers for classification
        # label_encoders = []
        # for i in range(targets.shape[-1]):
        #     le = LabelEncoder()
        #     targets[:, :, i] = le.fit_transform(targets[:, :, i].reshape(-1)).reshape(*targets.shape[:-1])
        #     label_encoders.append(le)
        # targets = targets.astype(int)

        # train/test indices
        # data_train, data_test, targets_train, targets_test = train_test_split(data, targets,
        #     train_size=train_size, shuffle=shuffle, random_state=seed)
        ix = np.arange(len(data))
        if shuffle:
            random_state = np.random.RandomState(seed=seed)
            ix = random_state.permutation(ix)
        split = int(train_size * len(data))
        ix_train = ix[:split]
        ix_test = ix[split:]

        # save the data
        savedir = Path(datapath).parent
        # np.savez("%s/train.npz" % savedir, data=data[ix_train], targets=targets[ix_train])
        # np.savez("%s/test.npz" % savedir, data=data[ix_test], targets=targets[ix_test])
        torch.save([data[ix_train], targets[ix_train]], "%s/train_stack%d.pt" % (savedir, img_stack))
        torch.save([data[ix_test], targets[ix_test]], "%s/test_stack%d.pt" % (savedir, img_stack))
