from setuptools import setup

setup(name='cgi_drl',
      version='0.2.0',
      install_requires=[
            'opencv-python',
            'moviepy',
            'numpy==1.23.4',
            'python-csv',
            'torch==1.13.1+cu116'
            'tensorboardX',
            'msgpack',
            'msgpack_numpy'
      ]
)
