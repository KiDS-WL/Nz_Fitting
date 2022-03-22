import setuptools


with open("README.md", "r") as f:
    long_description = f.read()
with open("requirements.txt") as f:
    install_requires = [pkg.strip() for pkg in f.readlines() if pkg.strip()]

setuptools.setup(
    name="Nz_Fitting",
    version="1.0",
    author="Jan Luca van den Busch",
    description="Repository for Nz fitting packages.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KiDS-WL/Nz_Fitting",
    packages=setuptools.find_packages(),
    install_requires=install_requires)
