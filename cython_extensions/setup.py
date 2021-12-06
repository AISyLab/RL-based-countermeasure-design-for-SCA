import setuptools
from Cython.Build import cythonize
import numpy

ext_options = {"compiler_directives": {"profile": False}, "annotate": False}
setuptools.setup(
    name="countermeasures_cython",
    version='0.0.1',
    packages=[
        'aisylab.cython_extensions.countermeasures_cython'
    ],
    package_dir={
        'aisylab.cython_extensions.countermeasures_cython': './countermeasures_cython'
    },
    ext_package='aisylab',
    ext_modules=cythonize(
        [
            "countermeasures_cython/clock_jitter.pyx",
            "countermeasures_cython/random_delay_interrupt.pyx"
        ],
        **ext_options
    ),
    include_dirs=[numpy.get_include()],
)
