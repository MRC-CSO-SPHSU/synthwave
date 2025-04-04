"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="synthwave",
    version="0.1.0",
    description="Population dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MRC-CSO-SPHSU/synthwave",
    author="Vladimir Khodygo",
    author_email="vladimir.khodygo@glasgow.ac.uk",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="simulation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.12",
    install_requires=["pyarrow>=19.0.0",
                      "tqdm>=4.66.5",
                      "pyyaml>=6.0.2",
                      "scikit-learn>=1.6.1",
                      "sdv>=1.17.3",
                      "torch>=2.5.1",
                      "matplotlib>=3.10.0",
                      "jupyter",
                      "seaborn",
                      ],
    extras_require={
        "dev": ["check-manifest"],
        "test": ["coverage"],
    },
    include_package_data=True,
    package_data={"synthwave": ["*.yaml", "*.yml"]}
)
