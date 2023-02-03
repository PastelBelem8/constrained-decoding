# coding: utf-8
## To check what packages are being included in the wheel:
## python -c "from setuptools import setup, find_packages; print(find_packages())"
"""Setup Sampling package."""

from setuptools import setup, find_packages

setup(
    name='sampling',
    version='0.0.1',
    description='Implementation of probabilistic predictive querying',
    author='Alex Boyd, Catarina Belem',
    author_email='catarina.garcia.belem@gmail.com',
    url='https://arxiv.org/abs/2210.06464',
    install_requires=[],
    packages=find_packages(),
    package_dir={"": "src"},
    package_data={},
)