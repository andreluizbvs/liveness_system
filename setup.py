from setuptools import setup, find_packages

setup(
    name='liveness_system',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'tensorflow',
        'keras',
        'matplotlib',
        'opencv-python'
    ],
)