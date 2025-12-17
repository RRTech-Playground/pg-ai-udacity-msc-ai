from setuptools import setup, find_packages

setup(
    name="ringgis_probability_distributions",
    version="0.2.7",
    description="Gaussian and Binomial distributions",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=['ringgis_probability_distributions'],
    author='Roland Ringgenberg',
    author_email='roland@rolandringgenberg.com',
    zip_safe=False,
)