import urllib.request
import os

from pydl_audio_encoders.library.utility.download_utils import reporthook


def download_cifar10_model_if_not_found(flag_file):
    if os.path.exists(flag_file):
        return

    if not os.path.exists(flag_file):
        url_link = 'https://www.dropbox.com/s/tr6f98vfwyefvtr/cifar10.pb?dl=1'
        print('pb model file does not exist, downloading from internet')
        urllib.request.urlretrieve(url=url_link, filename=flag_file,
                                   reporthook=reporthook)


