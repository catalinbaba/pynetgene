from setuptools import setup, find_packages

setup(
    name='pynetgene',                      # The name of your package
    version='1.2.0',                       # The current version of the package
    author="Catalin Baba",                 # The name of the package author
    author_email="catalin.viorelbaba@gmail.com",  # Email address of the author
    packages=find_packages(),              # Automatically find all packages (Python directories with an __init__.py file)
    install_requires=[                     # List of dependencies needed to run the package
        'pytest>=8.1.1',                   # Specify minimum version requirement for pytest
    ],
    description="PyNetgene: Python Genetic Algorithms Library"  # A short description of the package
)
