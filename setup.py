from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build this package")

        build_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
        os.makedirs(build_directory, exist_ok=True)
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(self.build_lib)}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]
        
        build_args = ['--config', 'Release', '--', '-j4']
        
        subprocess.check_call(['cmake', '..'] + cmake_args, cwd=build_directory)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_directory)

setup(
    name='qwen-tokenizer-fast',
    version='0.1.0',
    packages=find_packages(),
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)

