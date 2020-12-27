import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='sparselsh',
    version='2.0.2',
    author='Brandon Roberts',
    author_email='brandon@bxroberts.org',
    description='A locality sensitive hashing library with an emphasis on large, sparse datasets.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/brandonrobertz/sparselsh',
    download_url='https://github.com/brandonrobertz/SparseLSH/releases/tag/v2.0.2',
    keywords = ['clustering', 'sparse', 'lsh'],
    packages = setuptools.find_packages(),
    install_requires=[
        'numpy>=1.18.4,<2.0',
        'scipy>=1.4.1,<2.0'
    ],
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
