from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='deep_learning_projects',
      version='0.1',
      description="Guy Davidson's DL projects",
      url='https://github.com/guydav/deep-learning-projects',
      author='Guy Davidson',
      author_email='guy@minerva.kgi.edu',
      license='N/A',
      packages=['projects'],
      install_requires=requirements,
      zip_safe=True)