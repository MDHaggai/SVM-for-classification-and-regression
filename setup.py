from setuptools import setup, find_packages

setup(
    name='svm-classification-regression',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project implementing Support Vector Machines for classification and regression tasks.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/svm-classification-regression',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyter',
        'notebook',
        'pytest'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)