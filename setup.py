from setuptools import setup, find_packages, Extension, Command
from setuptools.command.egg_info import egg_info
from setuptools.command.build_ext import build_ext
import shutil
import glob
import os
import sys


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir + "src")


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            try:
                import cmake
            except ImportError:
                self.warn("cmake was not found, please install cmake `pip install cmake`")
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        os.makedirs(self.build_temp, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"
        cmake_cmd = [
            "cmake",
            "-S",
            ext.sourcedir,
            "-B",
            self.build_temp,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DCMAKE_BUILD_TYPE=" + cfg,
        ]

        if os.getenv("USE_PYTHON_3_11") == "ON":
            cmake_cmd.append("-DUSE_PYTHON_3_11=ON")

        self.announce(f"Configuring project", level=2)
        self.spawn(cmake_cmd)

        self.announce(f"Building project", level=2)
        self.spawn(["cmake", "--build", self.build_temp, "--config", cfg])

        # Directory where the shared library will be copied
        lib_dest_directory = "src/python/intel_visual_ai"
        os.makedirs(lib_dest_directory, exist_ok=True)

        # Path to the built shared library
        built_lib_path = os.path.join(self.build_temp, "lib/libvisual_ai.so")

        if not os.path.isfile(built_lib_path):
            raise FileNotFoundError(
                "libvisual_ai.so not found. Please ensure it is built correctly."
            )

        self.copy_file(built_lib_path, lib_dest_directory)
        self.announce(f"Copied libvisual_ai.so to {lib_dest_directory}")


class CustomEggInfo(egg_info):
    def run(self):
        self.run_command("build_ext")
        super().run()


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        directories = ["./build", "./dist", "./src/python/libvisual_ai.egg-info"]
        files = ["./src/python/intel_visual_ai/libvisual_ai.so"]

        for directory in directories:
            if os.path.exists(directory):
                print(f"Removing directory: {directory}")
                shutil.rmtree(directory)

        for file_pattern in files:
            for filepath in glob.glob(file_pattern):
                print(f"Removing file: {filepath}")
                os.remove(filepath)


setup(
    name="intel_visual_ai",
    version="0.5.0",
    package_dir={"": "src/python"},
    packages=find_packages("src/python"),
    ext_modules=[CMakeExtension("intel_visual_ai")],
    install_requires=["cmake"],
    cmdclass={
        "build_ext": CMakeBuild,
        "egg_info": CustomEggInfo,
        "clean": CleanCommand,
    },
    package_data={"intel_visual_ai": ["libvisual_ai.so"]},
)
