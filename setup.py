import setuptools

with open('requirements.txt') as f:
    requires = [
        r.split('/')[-1] if r.startswith('git+') else r
        for r in f.read().splitlines()]

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(
    name='linposfit',
    version='0.0.1',
    description='Linearized position reconstruction fitter',
    author='Auke-Pieter Colijn, Peter Gaemers',
    url='https://github.com/xams-nikhef/linposfit',
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    setup_requires=['pytest-runner'],
    install_requires=requires,
    tests_require=requires + ['pytest',
                              'hypothesis',
                              'flake8',
                              'pytest-cov',
                              ],
    python_requires=">=3.6",
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'nbsphinx',
            'recommonmark',
            'graphviz']},
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Physics'],
    zip_safe=False)