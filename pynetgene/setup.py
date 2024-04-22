from setuptools import setup, find_packages

setup(
    name='pynetgene',
    version='1.1.0',
    author="Catalin Baba",
    author_email="catalin.viorelbaba@gmail.com",
    packages=find_packages(),
    install_requires=[
        'pytest>=8.1.1',
    ],
    description="PyNetgene: A Python Library for Building the Genetic Algorithm"

)
