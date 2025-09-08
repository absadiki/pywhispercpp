"""Custom PEP 517 build backend that handles config settings for acceleration."""

import os
from setuptools import build_meta as _orig

# Re-export everything from setuptools
__all__ = _orig.__all__

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel with config_settings support for acceleration and repair options."""
    
    if config_settings:
        # Handle uv -C flag format: -Caccel=cuda, -Crepair=false, etc.
        for key, value in config_settings.items():
            if key == 'accel':
                # Handle acceleration options
                if isinstance(value, str):
                    value = [value]
                elif not isinstance(value, list):
                    value = [str(value)]
                
                for accel in value:
                    accel = accel.lower()
                    if 'cuda' in accel:
                        os.environ['GGML_CUDA'] = '1'
                        print("Enabling CUDA support via config-settings")
                    elif 'coreml' in accel:
                        os.environ['WHISPER_COREML'] = '1'
                        os.environ['WHISPER_COREML_ALLOW_FALLBACK'] = '1'
                        print("Enabling CoreML support via config-settings")
                    elif 'vulkan' in accel:
                        os.environ['GGML_VULKAN'] = '1'
                        print("Enabling Vulkan support via config-settings")
                    elif 'openblas' in accel or 'blas' in accel:
                        os.environ['GGML_BLAS'] = '1'
                        print("Enabling OpenBLAS support via config-settings")
            
            elif key == 'repair':
                # Handle repair option
                if str(value).lower() in ['false', '0', 'no', 'off']:
                    os.environ['NO_REPAIR'] = '1'
                    print("Disabling wheel repair via config-settings")
                elif str(value).lower() in ['true', '1', 'yes', 'on']:
                    os.environ.pop('NO_REPAIR', None)
                    print("Enabling wheel repair via config-settings")
        
        # Remove our custom options before passing to setuptools
        # setuptools doesn't understand them and will error
        cleaned_settings = {k: v for k, v in config_settings.items() 
                          if k not in ['accel', 'repair']}
        config_settings = cleaned_settings if cleaned_settings else None
    
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)

# Re-export other functions
build_sdist = _orig.build_sdist
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist