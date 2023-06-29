from setuptools import setup, find_packages

setup(
    name='slicr',
    version='0.1.5',
    description='SLICR: Sparse Locally Involved Covariate Regression. A Python package for efficient, memory-conscious kNN distance correction, based on technical covariates whose effect should be removed.',
    author='Scott Tyler',
    author_email='scottyler89@gmail.com',
    url='https://github.com/scottyler89/slicr',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'scikit-learn',
        'networkx',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python :: 3.8',
    ],
)

