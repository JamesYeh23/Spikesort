from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nex-sdk-jy',
    version='0.1.0',
    packages=['nex_sdk'],
    # packages=find_packages()
    install_requires=requirements,  
    description='A Python toolkit for working with Nex files',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='James Yeh',
    author_email='you@example.com',
    url='https://github.com/JamesYeh23/Spikesort',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)