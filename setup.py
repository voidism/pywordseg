import setuptools
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
  name="pywordseg",
  version="0.1.2",
  author="Jexus Chuang",
  description="Open source state-of-the-art Chinese word segmentation toolkit",
  long_description=long_description,
  long_description_content_type='text/markdown',
  url="https://github.com/voidism/pywordseg",
  packages=setuptools.find_packages(),
  license='MIT',
  install_requires=[
    "torch",
    "h5py",
    "numpy",
    "overrides",
  ],
  classifiers=[
    "Development Status :: 2 - Pre-Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
  ],
  exclude_package_date={'':['.gitignore','.git','models','ELMoForManyLangs','CharEmb']}
)
