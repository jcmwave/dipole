from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dipole", 
    version="0.0.0-alpha.1",
    author="Phillip Manley",
    author_email="phillip.manley@jcmwave.com",
    description="utility functions for dipole simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'matplotlib', 'scipy'],
    python_requires='>=3.9',
    include_package_data=True,
)
