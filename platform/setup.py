from setuptools import setup

setup(name='cgi_drl',
      version='0.2.0',
      install_requires=[
            'setuptools<58.0.0',
            'pyyaml==6.0',
            'gym==0.12.0',
            'atari-py==0.2.9',
            'opencv-python',
            'moviepy',
            'numpy==1.23.4',
            'python-csv',
            'torch==1.13.1+cu116'
      ]
)
