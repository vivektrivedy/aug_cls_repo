from setuptools import setup, find_packages
setup(
    name="cls-reg-roi-retrieval",
    version="0.0.3",
    description="Multi-vector ViT retrieval (CLS+Reg+ROI)",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[r.strip() for r in open("requirements.txt")],
    python_requires=">=3.9",
)
