from setuptools import setup, find_packages


with open('README.md', encoding="UTF-8") as f:
    readme = f.read()

setup(
    name='baumgarte',
    version='0.1.0',
    description='Simple physics engine.',
    long_description=readme,
    author='kzkr19',
    url='https://gitlab.com/kazukuro19/Baumgarte2D',
    packages=find_packages(exclude=('tests', 'docs')),
)
