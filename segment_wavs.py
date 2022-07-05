import sounddevice as sd
import numpy as np
import librosa
import os

librispeech_root = "/home/praveen/projects/librispeech_alignments/LibriSpeech"
target_dir = "/home/praveen/projects/librispeech_alignments/librispeech_segmented"
sr = 16000

def create_target_dir_structure():
    for set_name in os.listdir(librispeech_root):
        set_dir = os.path.join(librispeech_root, set_name)

        if not os.path.isdir(set_dir):
            continue

        if not os.path.exists(os.path.join(target_dir, set_name)):
            os.makedirs(os.path.join(target_dir, set_name))

        target_set_dir = os.path.join(target_dir, set_name)
            
        for spkr_id in os.listdir(set_dir):
            spkr_dir = os.path.join(set_dir, spkr_id)
            target_spkr_dir = os.path.join(target_set_dir, spkr_id)

            for book_id in os.listdir(spkr_dir):
                book_dir = os.path.join(spkr_dir, book_id)

                target_book_dir = os.path.join(target_spkr_dir, book_id)

                if not os.path.exists(os.path.join(target_book_dir)):
                    os.makedirs(target_book_dir)

def segment_utterance(audio_fpath, utterance_id, words, end_times):
    start_times = [0] + end_times[:-1]
    audio_file, _ = librosa.load(audio_fpath, sr)
    segments = []

    
    for i in range(len(words)):
        start = start_times[i]
        end = end_times[i]
        text = words[i]
        segment = audio_file[int(start * sr):int(end * sr)]

        print('text', text)
        print('segment length', len(segment))
        #sd.play(segment, sr, blocking = True)

        segments.append(segment)

    return segments


def write_data(target_book_dir, utterance_id, words, segments):
    
    assert len(words) == len(segments)
    


    for i, (word, segment) in enumerate(zip(words, segments)):

        utterance_id_path = os.path.join(target_book_dir, utterance_id)
        with open(utterance_id_path + '_' + str(i) + '.txt', 'w') as wordfile:
            wordfile.write(word)

        wordfile.close()
        librosa.output.write_wav(utterance_id_path + '_' + str(i) + '.wav', segment, sr)



def process_dataset():
    for set_name in os.listdir(librispeech_root):
        set_dir = os.path.join(librispeech_root, set_name)

        if not os.path.isdir(set_dir):
            continue


        target_set_dir = os.path.join(target_dir, set_name)

        for spkr_id in os.listdir(set_dir):
            spkr_dir = os.path.join(set_dir, spkr_id)
            target_spkr_dir = os.path.join(target_set_dir, spkr_id)

            for book_id in os.listdir(spkr_dir):
                book_dir = os.path.join(spkr_dir, book_id)

                target_book_dir = os.path.join(target_spkr_dir, book_id)

                alignment_fpath = os.path.join(book_dir, "%s-%s.alignment.txt" %
                                               (spkr_id, book_id))
                if not os.path.exists(alignment_fpath):
                    raise Exception("Alignment file not found. Did you download and merge the txt "
                                    "alignments with your LibriSpeech dataset?")

                alignment_file = open(alignment_fpath, "r")

                for line in alignment_file:
                    utterance_id, words, end_times = line.strip().split(' ')
                    words = words.replace('\"', '').split(',')
                    end_times = [float(e) for e in end_times.replace('\"', '').split(',')]
                    audio_fpath = os.path.join(book_dir, utterance_id + '.flac')

                    segments = segment_utterance(audio_fpath, utterance_id, words, end_times)

                    write_data(target_book_dir, utterance_id, words, segments)


                    
                
                
if __name__ == '__main__':
    create_target_dir_structure()
    print('created_target_structure')
    process_dataset()
    
    
    
            
            
        
    
