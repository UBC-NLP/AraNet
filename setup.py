from setuptools import setup

setup(name='aranet',
      version='0.1',
      description='description',
      url='https://github.com/UBC-NLP/aranet',
      author='',
      author_email='',
      license='GNU',
      packages=['aranet'],
      install_requires=[
          'tensorflow-gpu',
          'torch',
          'sklearn',
          'transformers==4.30.0', 'keras', 'pandas', 'numpy'
      ],

      entry_points={
          'console_scripts': [
              'aranet = aranet.aranet:main',
          ],
      },
      package_data={'aranet': ['resources/*']},
      include_package_data=True,
      zip_safe=False)
