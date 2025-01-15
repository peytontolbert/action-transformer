from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="action-transformer",
    version="0.1.0",
    author="Action Transformer Contributors",
    author_email="",
    description="A Transformer-based model for sequential decision-making in RL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peytontolbert/action-transformer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
            "mypy>=0.9.0",
        ],
        "examples": [
            "gym>=0.21.0",
            "pettingzoo>=1.22.0",
            "stable-baselines3>=1.7.0",
        ],
    },
) 