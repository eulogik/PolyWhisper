from setuptools import setup, find_packages

setup(
    name="polywhisper",
    version="0.2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "numpy",
        "soundfile",
        "resampy",
    ],
    entry_points={
        "console_scripts": [
            "polywhisper=polywhisper.cli:main",
        ],
    },
    description="Efficient multilingual Indic ASR via frozen Whisper + per-language LoRA",
    author="PolyWhisper Team",
)
