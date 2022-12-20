import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='sparselsh',
    version='2.1.1',
    author='Brandon Roberts',
    author_email='brandon@bxroberts.org',
    description='A locality sensitive hashing library with an emphasis on large, sparse datasets.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/brandonrobertz/sparselsh',
    download_url='https://github.com/brandonrobertz/SparseLSH/releases/tag/v2.1.1',
    keywords=['clustering', 'sparse', 'lsh'],
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.18.6,<2.0',
        'scipy>=1.0.0,<2.0',
        'scikit-learn>=0.24.0,<2.0'
    ],
    extras_require={
        "test": ["pytest"],
        "redis": ["redis>=2.10.1,<3.0"]
    },
    entry_points="""
        [console_scripts]
        sparselsh=sparselsh.cli:cli
    """,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
        ],
)
