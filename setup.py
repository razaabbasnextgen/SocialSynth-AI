#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from setuptools import find_packages, setup

# Package meta-data
NAME = 'socialsynth-ai'
DESCRIPTION = 'Advanced content generation tool leveraging knowledge graphs and AI'
URL = 'https://github.com/razaabbasnextgen/SocialSynth-AI/'
EMAIL = 'razaabbas2529@gmail.com'
AUTHOR = 'Raza Abbas'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = '1.0.0'

# Required packages
REQUIRED = [
    'streamlit>=1.31.0',
    'networkx>=3.1',
    'pyvis>=0.3.2',
    'pandas>=2.0.0',
    'matplotlib>=3.7.1',
    'python-dotenv>=1.0.0',
    'spacy>=3.6.0',
    'scikit-learn>=1.3.0',
    'sentence-transformers>=2.2.2',
    'langchain>=0.1.0',
    'langchain-community>=0.0.10',
    'langchain-core>=0.1.5',
    'chromadb>=0.4.18',
    'PyPDF2>=3.0.0',
    'google-api-python-client>=2.100.0',
    'google-generativeai>=0.3.0',
]

# Optional packages
EXTRAS = {
    'dev': [
        'pytest>=7.4.0',
        'black>=23.7.0',
        'mypy>=1.5.1',
        'isort>=5.12.0',
    ],
    'docs': [
        'sphinx>=7.1.0',
        'sphinx-rtd-theme>=1.3.0',
    ],
}

# Import the README
here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    entry_points={
        'console_scripts': [
            'socialsynth=src.socialsynth.ui.streamlit_app:main',
        ],
    },
) 