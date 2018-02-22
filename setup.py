from setuptools import setup

setup(
    name='pydl_audio_encoders',
    packages=['pydl_audio_encoders'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
        'sklearn',
        'nltk',
        'numpy',
        'h5py'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)