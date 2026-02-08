from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="emotion-ai",
    version="1.0.0",
    author="M Saivishnu Sarode",
    author_email="saivishnusarode@gmail.com",
    description="Dual-Model Facial Expression Recognition System using ResNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/emotion-ai-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.9b0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "emotion-ai-train=src.training.train_emotions:main",
            "emotion-ai-serve=api.app:main",
        ],
    },
)
