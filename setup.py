from setuptools import setup, find_packages

setup(
    name='forwardThinking',
    version='0.0.1',
    url='https://github.com/blackecho/Deep-Learning-Tensorflow',
    download_url='https://github.com/blackecho/Deep-Learning-TensorFlow/tarball/0.0.6',
    author='Sean Wade',
    author_email='seanwademail@gmail.com',
    description='An implementation of variations of forwardThinking models.',
    packages=find_packages(exclude=['tests']),
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    license='MIT',
    install_requires=[],
)
