from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='carte_territoire_package',
      version="0.1",
      description="Projet carte et territoire",
      install_requires=requirements,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
