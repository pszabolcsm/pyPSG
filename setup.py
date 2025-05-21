from setuptools import setup
import os

# Parse README.md as long_description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Parse requirements.txt as install_requires
with open("requirements.txt", "r", encoding="utf-8") as f:
    require = f.read().splitlines()

setup(
    name="pyPSG", # Name of the package
    version="1.0.0",
    description="Python toolbox for PPG, ECG, SPO2 and HRV analysis",
    author="Szabolcs M. Péter, Marton A. Goda, PhD",
    author_email="peter.szabolcs.matyas@hallgato.ppke.hu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pszabolcsm/pyPSG",
    project_urls={"Bug Tracker": "http://https://github.com/pszabolcsm/pyPSG/issues",},
    classifiers=[
        "Development Status :: 4 - Implementation",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    packages=["pyPSG", "pyPSG"+os.sep+"biomarkers","pyPSG"+os.sep+"IO"], # Name of the package directory
    install_requires=[require],
    python_requires=">=3.10"
)