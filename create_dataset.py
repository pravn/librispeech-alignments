import torch
import numpy as np
import os
import glob

librispeech_root = "/home/praveen/projects/librispeech_alignments/LibriSpeech"
segments_dir = "/home/praveen/projects/librispeech_alignments/librispeech_segmented"
set_name = 'train-clean-100'
mel_files_list = '/home/praveen/projects/librispeech_alignments/mels_list.txt'
audio_files_list = '/home/praveen/projects/librispeech_alignments/audio_list.txt'

set_dir = os.path.join(librispeech_root, set_name)
segments_set_dir = os.path.join(segments_dir, set_name)

mel_files = []
audio_files = []

for spkr_id in os.listdir(set_dir):
    spkr_dir = os.path.join(set_dir, spkr_id)
    segments_spkr_dir = os.path.join(segments_set_dir, spkr_id)
    
    for book_id in os.listdir(spkr_dir):
        book_dir = os.path.join(spkr_dir, book_id)

        segments_book_dir = os.path.join(segments_spkr_dir, book_id)

        alignment_fpath = os.path.join(book_dir, "%s-%s.alignment.txt" %
                                        (spkr_id, book_id))

        if not os.path.exists(alignment_fpath):
            raise Exception("Alignment file not found. Did you download and merge the txt "
                            "alignments with your LibriSpeech dataset?")

        
        alignment_file = open(alignment_fpath, "r")

        for line in alignment_file:
            utterance_id, _, _ = line.strip().split(' ')
            mel_segments_fpath = glob.glob(os.path.join(segments_book_dir, utterance_id + '*.npy'))
            audio_segments_fpath = glob.glob(os.path.join(segments_book_dir, utterance_id + '*.wav'))

            #print('mel_segments_fpath', mel_segments_fpath)

            for mel in mel_segments_fpath:
                mel_files.append(mel)

            for audio_file in audio_segments_fpath:
                audio_files.append(audio_file)

            #break

#print(mel_files[:10])

    
with open(mel_files_list, 'w') as mel_files_handle:
    for mel in mel_files:
        mel_files_handle.write(mel+'\n')

    mel_files_handle.close()

with open(audio_files_list, 'w') as audio_files_handle:
    for audio_file in audio_files:
        audio_files_handle.write(audio_file+'\n')

    audio_files_handle.close()

    










            







