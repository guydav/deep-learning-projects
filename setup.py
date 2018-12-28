from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

requirements = list(filter(lambda s: not s.startswith('-e'), requirements))

setup(name='projects',
      version='0.1',
      description="Guy Davidson's DL projects",
      url='https://github.com/guydav/deep-learning-projects',
      author='Guy Davidson',
      author_email='guy@minerva.kgi.edu',
      license='N/A',
      packages=find_packages(exclude=['contrib', 'docs', 'tests', 'notebooks']),
      install_requires=requirements)
