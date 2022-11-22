from setuptools import setup


setup(
    name='cyjax',
    version='1.0.0',
    description='Numerical methods for Calabi-Yau metrics in JAX',
    author='Mathis Gerdes',
    author_email='MathisGerdes@gmail.com',
    packages=['cyjax'],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'sympy',
        'jax',
        'jaxlib',
        'chex',
        'progeval',
        'jax-autovmap',
        'matplotlib',
        'flax',
        'optax',
    ],
)