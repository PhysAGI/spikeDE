from setuptools import setup, find_packages

setup(
    name="spikeDE",
    author="Qiyu Kang",
    author_email="qiyukang@ustc.edu.cn",
    description="SNN in PyTorch with the adjoint backpropagation technique.",
    url="https://github.com/kangqiyu/spikeDE",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.7",
)
