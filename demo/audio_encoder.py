from random import shuffle
import os
import sys


def load_audio_path_label_pairs(max_allowed_pairs=None):
    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir is not '' else '.'

    from pydl_audio_encoders.library.utility.gtzan_loader import download_gtzan_genres_if_not_found
    download_gtzan_genres_if_not_found(current_dir + '/very_large_data/gtzan')
    audio_paths = []
    with open(current_dir + '/data/lists/test_songs_gtzan_list.txt', 'rt') as file:
        for line in file:
            audio_path = current_dir + '/very_large_data/' + line.strip()
            if max_allowed_pairs is None or len(audio_paths) < max_allowed_pairs:
                audio_paths.append(audio_path)
    pairs = []
    with open(current_dir + '/data/lists/test_gt_gtzan_list.txt', 'rt') as file:
        for line in file:
            label = int(line)
            if max_allowed_pairs is None or len(pairs) < max_allowed_pairs:
                pairs.append((audio_paths[len(pairs)], label))
            else:
                break
    return pairs


def main():
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    audio_path_label_pairs = load_audio_path_label_pairs(1)

    from pydl_audio_encoders.library.cifar10 import Cifar10AudioEncoder
    encoder = Cifar10AudioEncoder()

    audio_path, actual_label_id = audio_path_label_pairs[0]
    encoded_audio = encoder.encode(audio_path)

    print(encoded_audio)


if __name__ == '__main__':
    main()
