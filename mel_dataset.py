from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class MelDataset(Dataset):
    def __init__(self, mels_fpath, audio_fpath=None, count=None):
        self.mels_fpath= read_mel_files_list(mels_fpath, count)
        self.audio_fpath = audio_fpath
        self.pad_length = 96
        self.count = count
        print('self.mels_fpath.size', len(self.mels_fpath))

    def __len__(self):
        return len(self.mels_fpath)

    def __getitem__(self, idx):
        #mel = read_mels(self.mels_fpath[idx])
        mel = np.load(self.mels_fpath[idx])
        #print('mel.shape[1] h', mel.shape[1])
        mel = self.pad_mels(mel)
        #mels = torch.from_numpy(mels)
        return mel
            
    def pad_mels(self, mel):
        padded = np.zeros((mel.shape[0], self.pad_length))
        #print('s',mel.shape[1] < self.pad_length)
        if mel.shape[1] < self.pad_length:
            padded[:, :mel.shape[1]] = mel
        else:
            padded = mel[:,:padded.shape[1]]
            #print('mel.shape1', mel.shape[1])

        padded = torch.from_numpy(padded)
        return padded

def read_mel_files_list(mels_fpath, count):
    with open(mels_fpath, 'r') as mel_files_handle:
        mels_fpath = [entry.strip('\n') for entry in mel_files_handle]

    mel_files_handle.close()

    if count is not None:
        return mel_fpath[:count]

    return mels_fpath


def read_mels(mels_fpath):
    mels = []
    for mel_file in mels_fpath:
        mels.append(np.load(mel_file))
    return mels



#mel_files = '/home/praveen/projects/librispeech_alignments/mels_list.txt'
#mel_dataset = MelDataset(mel_files)
        




        