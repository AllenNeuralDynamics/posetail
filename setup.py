from setuptools import setup, find_packages

setup(
    name = 'posetail', 
    version = '0.8.0',
    description = 'a model for tracking 2d or 3d animal pose through time',  
    url = 'https://github.com/AllenNeuralDynamics/posetail', 
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
    ],
    keywords = '3d point tracking, 2d point tracking, multi-view tracking, pose estimation',
    packages = find_packages(), 
)