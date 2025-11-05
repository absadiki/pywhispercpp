# This setup.py is used to build the pywhispercpp package.
# The environment variables you may find interesting are:
#
# PYWHISPERCPP_VERSION 
# if set, it will be used as the version number.
#
# GGML_VULKAN=1
# if set, whisper.cpp will be build with vulkan support.
#
# WHISPER_COREML=1
# WHISPER_COREML_ALLOW_FALLBACK=1
# if set, whisper.cpp will be build with coreml support which requires special models
# It is best used with WHISPER_COREML_ALLOW_FALLBACK=1


import os
import re
import subprocess
import sys
from pathlib import Path
import subprocess

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_wheel import bdist_wheel

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


dll_folder = 'unset'


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython_EXECUTABLE={sys.executable}",  # Alternative Python executable setting
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        if self.editable_mode:
            # Platform-specific rpath settings
            if sys.platform.startswith('darwin'):
                # macOS-specific settings
                cmake_args += [
                    "-DCMAKE_INSTALL_RPATH=@loader_path",
                    "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
                ]
            elif sys.platform.startswith('linux'):
                # Linux-specific settings
                cmake_args += [
                    "-DCMAKE_INSTALL_RPATH=$ORIGIN",
                    "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
                ]

        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]  # type: ignore[attr-defined]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)
        else:
            # Clean up any corrupted CMake cache files
            cmake_cache = build_temp / "CMakeCache.txt"
            if cmake_cache.exists():
                try:
                    cmake_cache.unlink()
                    print("Removed corrupted CMakeCache.txt")
                except Exception as e:
                    print(f"Warning: Could not remove CMakeCache.txt: {e}")

        # Only pass specific environment variables that are safe for CMake
        # and relevant to the build process
        safe_env_vars = [
            'GGML_CUDA', 'GGML_VULKAN', 'GGML_BLAS', 'GGML_OPENBLAS',
            'WHISPER_COREML', 'WHISPER_COREML_ALLOW_FALLBACK',
            'CMAKE_BUILD_PARALLEL_LEVEL', 'CMAKE_GENERATOR',
            'CMAKE_ARGS', 'DEBUG', 'VERBOSE'
        ]
        
        for key, value in os.environ.items():
            # Only pass safe environment variables that don't contain problematic characters
            if (key in safe_env_vars and 
                value and 
                '\n' not in value and 
                '\r' not in value and
                not key.startswith('FZF_') and
                not key.startswith('PS1') and
                not key.startswith('PROMPT')):
                cmake_args.append(f'-D{key}={value}')

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )

        # store the dll folder in a global variable to use in repairwheel
        global dll_folder
        cfg = "Debug" if self.debug else "Release"
        dll_folder = os.path.join(self.build_temp, '_pywhispercpp', 'bin', cfg)
        print("dll_folder in build_extension", dll_folder)
        # self.copy_extensions_to_source()

    def copy_extensions_to_source(self):
        super().copy_extensions_to_source()

        if self.editable_mode:
            build_lib = Path(self.build_lib)
            for ext in self.extensions:
                extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
                # Assuming all shared libraries are in the same directory
                shared_lib_files = [*build_lib.glob('**/*.dylib'), *build_lib.glob('**/*.so*')]
                for shared_lib in shared_lib_files:
                    self.copy_file(shared_lib, extdir)


# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


class RepairWheel(bdist_wheel):
    def run(self):
        super().run()
        if os.environ.get('NO_REPAIR', '0') == '1':
            print("Skipping wheel repair")
            return
        if os.environ.get('CIBUILDWHEEL', '0') == '0' or sys.platform.startswith('win'):
            # for linux and macos we use the default wheel repair command from cibuildwheel, for windows we need to do it manually as there is no repair command
            self.repair_wheel()

    def repair_wheel(self):
        # on windows the dlls are in D:\a\pywhispercpp\pywhispercpp\build\temp.win-amd64-cpython-311\Release\_pywhispercpp\bin\Release\whisper.dll
        global dll_folder
        print("dll_folder in repairwheel", dll_folder)
        print("Files in dll_folder:", *Path(dll_folder).glob('*'))
        # build\temp.win-amd64-cpython-311\Release\_pywhispercpp\bin\Release\whisper.dll

        wheel_path = next(Path(self.dist_dir).glob(f"{self.distribution.get_name()}*.whl"))
        # Create a temporary directory for the repaired wheel
        import tempfile
        with tempfile.TemporaryDirectory(prefix='repaired_wheel_') as tmp_dir:
            tmp_dir = Path(tmp_dir)
            subprocess.call(['repairwheel', wheel_path, '-o', tmp_dir, '-l', dll_folder])
            print("Repaired wheel: ", *tmp_dir.glob('*.whl'))
            # We need to glob as repairwheel may change the name of the wheel 
            # on linux from pywhispercpp-1.2.0-cp312-cp312-linux_aarch64.whl 
            #            to pywhispercpp-1.2.0-cp312-cp312-manylinux_2_34_aarch64.whl
            repaired_wheel = next(tmp_dir.glob("*.whl"))
            self.copy_file(repaired_wheel, wheel_path)
            print(f"Copied repaired wheel to: {wheel_path}")


def get_local_version() -> str:
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        return f"+git{git_sha[:7]}"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def get_version() -> str:
    try:
        return os.environ['PYWHISPERCPP_VERSION']
    except KeyError:
        pass
    with open("version.txt") as f:
        version = f.read().strip()
    return f"{version}{get_local_version()}"


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="pywhispercpp",
    author="absadiki",
    description="Python bindings for whisper.cpp",
    long_description=long_description,
    ext_modules=[CMakeExtension("_pywhispercpp")],
    cmdclass={"build_ext": CMakeBuild,
              'bdist_wheel': RepairWheel, },
    zip_safe=False,
    # extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.8",
    packages=find_packages('.'),
    package_dir={'': '.'},
    include_package_data=True,
    package_data={'pywhispercpp': []},
    long_description_content_type="text/markdown",
    license='MIT',
    entry_points={
        'console_scripts': ['pwcpp=pywhispercpp.examples.main:main',
                            'pwcpp-assistant=pywhispercpp.examples.assistant:_main',
                            'pwcpp-livestream=pywhispercpp.examples.livestream:_main',
                            'pwcpp-recording=pywhispercpp.examples.recording:_main',
                            'pwcpp-gui=pywhispercpp.examples.gui:_main', ]
    },
    project_urls={
        'Documentation': 'https://absadiki.github.io/pywhispercpp/',
        'Source': 'https://github.com/absadiki/pywhispercpp',
        'Tracker': 'https://github.com/absadiki/pywhispercpp/issues',
    },
    install_requires=['numpy', "requests", "tqdm", "platformdirs"],
    extras_require={"examples": ["sounddevice", "webrtcvad"],
                    "gui": ["pyqt5"]},
)
