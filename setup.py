from setuptools import setup

setup(name='xval',
    version='0.9',
    description='Extreme Value Estimates with Python',
    url='',
    author='Mathias Hauser',
    author_email='mathias.hauser@env.ethz.ch',
    license='MIT',
    packages=['xval'],
    install_requires=[
        'numpy',
		'scipy',
		'numdifftools',
		'matplotlib',
		],
    zip_safe=False)







