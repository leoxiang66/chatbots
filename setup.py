from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["torch",'transformers'] # 这里填依赖包信息

setup(
    name="chatbots",
    version="0.0.1",
    author="Tao Xiang",
    author_email="tao.xiang@tum.de",
    description="A package of chatbots",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/leoxiang66/chatbots",
    packages=find_packages(),
    # Single module也可以：
    # py_modules=['timedd']
    install_requires=requirements,
    classifiers=[
  "Programming Language :: Python :: 3.8",
  "License :: OSI Approved :: MIT License",
    ],
)