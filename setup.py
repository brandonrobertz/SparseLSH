# -*- coding: utf-8 -*-

import sparselsh

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('CHANGES.md') as f:
    changes = f.read()

required = ['numpy', 'scipy']

setup(
    name='sparselsh',
    version=sparselsh.__version__,
    packages=['sparselsh'],
    author='Brandon Roberts',
    author_email='brandon@bxroberts.org',
    maintainer='Brandon Roberts',
    maintainer_email='brandon@bxroberts.org',
    description='A locality sensitive hashing library with an emphasis on large (sparse) datasets.',
    long_description=readme + '\n\n' + changes,
    license=license,
    requires=required,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Topic :: Software Development :: Libraries',
        ],
)
