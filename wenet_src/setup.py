import platform
from setuptools import setup, find_packages

requirements = [
    "numpy==1.26.4",
    "requests",
    "tqdm",
    "torch==2.2.2",
    "torchaudio==2.2.2",
    "openai-whisper",
    "librosa",
    "pyyaml",
    "jieba",
    "langid",
    "sentencepiece",
    "deepspeed",
    "tensorboardX",
    "huggingface_hub"
]

extra_require = {
    "torch-npu": [
        "torch==2.2.0", "torch-npu==2.2.0", "torchaudio==2.2.0", "decorator",
        "numpy<2.0.0", "attrs", "psutil"
    ],
}

if platform.system() == 'Windows':
    requirements += ['PySoundFile']

setup(
    name="wenet",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={"console_scripts": [
        "wenet = wenet.cli.transcribe:main",
    ]},
    extras_require=extra_require,
)
