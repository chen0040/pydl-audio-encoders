# pydl-audio-encoders
Audio encoders that convert audios into numpy arrays for machine learning

The implementation of the audio is in the [pydl_audio_encoders/library](pydl_audio_encoders/library)

# Usage

### Audio Classifier

The [sample codes](demo/audio_classifier.py) shows how to use the encoder to predict the genre
of a song:

```python
from random import shuffle
import os
import sys


def load_audio_path_label_pairs(max_allowed_pairs=None):
    from pydl_audio_encoders.library.utility.gtzan_loader import download_gtzan_genres_if_not_found
    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir is not '' else '.'

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
    current_dir = current_dir if current_dir is not '' else '.'

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
```

### Audio Encoder

The [sample codes](demo/audio_encoder.py) shows how to use the encoder to encode varied-length
of audio file into fixed length numpy array:

```python
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

```

The above code encode a varied-length audio file to a numpy array of shape(512, )

In some cases, a low dimension encoding is desired, to do this change the following line

```python
encoder = ...
audio_path = ...
encoder.encode(audio_path)
```

to the following:

```python
encoder = ...
audio_path = ...
encoder.encode(audio_path, high_dimension=False)
```

This will encode a varied-length audio file to a numpy array of shape (10, )

# Note

### audioread.NoBackend

The audio processing depends on librosa version 0.6 which depends on audioread.  

If you are on Windows and sees the error "audioread.NoBackend", go to [ffmpeg](https://ffmpeg.zeranoe.com/builds/)
and download the shared linking build, unzip to a local directory and then add the bin folder of the 
ffmpeg to the Windows $PATH environment variable. Restart your cmd or powershell, Python should now be
able to locate the backend for audioread in librosa
