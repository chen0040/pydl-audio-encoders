from random import shuffle
import os
import sys


def load_audio_path_label_pairs(max_allowed_pairs=None):
    from pydl_audio_encoders.library.utility.gtzan_loader import download_gtzan_genres_if_not_found
    current_dir = os.path.dirname(__file__)
    download_gtzan_genres_if_not_found(current_dir + '/very_large_data/gtzan')
    audio_paths = []
    with open(current_dir + '/data/lists/test_songs_gtzan_list.txt', 'rt') as file:
        for line in file:
            audio_path = current_dir + '/very_large_data/' + line.strip()
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

    from pydl_audio_encoders.library.cifar10 import Cifar10AudioEncoder
    from pydl_audio_encoders.library.utility.gtzan_loader import gtzan_labels

    audio_path_label_pairs = load_audio_path_label_pairs()
    shuffle(audio_path_label_pairs)
    print('loaded: ', len(audio_path_label_pairs))

    encoder = Cifar10AudioEncoder()

    for i in range(0, 20):
        audio_path, actual_label_id = audio_path_label_pairs[i]
        predicted_label_id = encoder.predict_class(audio_path)
        actual_label = gtzan_labels[actual_label_id]
        predicted_label = gtzan_labels[predicted_label_id]

        print('Audio: ', audio_path)
        print('Predicted: ', predicted_label, 'Actual: ', actual_label)


if __name__ == '__main__':
    main()
