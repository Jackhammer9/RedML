from setuptools import setup, find_packages

setup(
    name='RedML',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    author='Arnav Bajaj',
    author_email='arnavbajaj9@gmail.com',
    description='The ultimate Python package machine learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Jackhammer9/RedML',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
