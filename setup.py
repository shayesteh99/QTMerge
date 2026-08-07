from setuptools import Extension, setup
from Cython.Build import cythonize


setup(
    name="qtmerge-fast",
    ext_modules=cythonize(
        [
            Extension(
                "qtmerge_fast",
                ["qtmerge_fast.pyx"],
                extra_compile_args=["-O3"],
            )
        ],
        language_level=3,
    ),
)
