from setuptools import setup, find_packages

setup(
    name='FluxPy',
    version='0.01',
    author='Jake Schaefer',
    author_email='schaeferj@carleton.edu',
    description='Large eddy simulation (LES) for simulating turbulence',
    license='',
    classifiers=(
        ""
    ),
    namespace_packages=['FluxPy'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'h5py',
    ],
    python_requires='>=3.8',
)
