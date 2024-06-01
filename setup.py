from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='ga_scheduler',
    version='2.0.2',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/ga_scheduler',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy'
    ],
    description='A Comprehensive Library for Solving Machine Scheduling Problems Using Genetic Algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
