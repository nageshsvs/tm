
from setuptools import setup

setup(name='timeseries_nag',
      version='0.1',
      description='The funniest joke in the world',
      url='https://github.com/nageshsvs/timeseries/blob/master/time_series_nagesh.py',
      dependency_links=['https://github.com/nageshsvs/timeseries/blob/master#egg=package-1.0'],
      author='Nagesh Somayajula',
      author_email='Nagesh.Somayajula@hitaciVantara.com',
      license='EDP',
      packages=['TimeSeries'],
      install_requires=[
          'statsmodels','sklearn',
      ],
      zip_safe=False)
