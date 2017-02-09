import sparselsh

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

install_requires = ['numpy==1.8.1', 'scipy==0.14.0']

setup(
    name='sparselsh',
    version=sparselsh.__version__,
    packages=['sparselsh'],
    author='Brandon Roberts',
    author_email='brandon@bxroberts.org',
    description='A locality sensitive hashing library with an emphasis on large (sparse) datasets.',
    url='https://github.com/brandonrobertz/sparselsh',
    download_url='https://github.com/brandonrobertz/sparselsh/tarball/v1.1.1',
    keywords = ['clustering', 'sparse', 'lsh'],
    install_requires=install_requires,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
        ],
)
