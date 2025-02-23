from setuptools import setup, find_packages

setup(
    name='DeepRIPE-Simulation',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-optimize",  # if you use skopt for parameter search
    ],
    entry_points={
        'console_scripts': [
            'dvripe_sim=src.simulation:run_dvripe_sim',  # optional, if you want a command-line entry point
        ],
    },
)
