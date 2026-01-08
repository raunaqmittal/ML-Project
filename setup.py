from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:  
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    requirements = [req.replace('\n', '') for req in requirements]
    if('-e .' in requirements):
        requirements.remove('-e .')
    return requirements

setup(
    name="ml project",
    version="0.0.1",
    author = 'Raunaq',
    author_email = 'raunaqmittal2004@gmail.com',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)