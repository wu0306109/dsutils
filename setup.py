from setuptools import find_packages, setup

with open('README.md') as stream:
    long_description = stream.read()

setup(
    name='dsutil',
    version='0.1.0',
    description='dsutil is a Python package containing useful functions for data science projects.',
    author='wu0306109',
    author_email='t110598007@ntut.org.tw',
    url='https://github.com/wu0306109/dsutils',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    requires=[
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
)
