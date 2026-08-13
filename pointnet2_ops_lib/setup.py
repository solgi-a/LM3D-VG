import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

requirements = ["torch>=1.4"]

exec(open(osp.join("pointnet2_ops", "_version.py")).read())

# GPU archs to build for. An explicit TORCH_CUDA_ARCH_LIST always wins; otherwise pick a
# default from the detected CUDA version (CUDA 12 dropped Kepler/compute_37 support, so the
# old hardcoded "3.7+PTX;..." list fails on modern toolkits).
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        import torch

        cuda_version = torch.version.cuda or ""
        major = int(cuda_version.split(".")[0]) if cuda_version else 0
    except Exception:
        major = 0

    if major >= 12:
        # Kepler (3.x) removed; Maxwell (5.x) deprecated. Pascal and newer only.
        # 6.0/6.1 = P100/T4-era, 7.0 = V100, 7.5 = T4, 8.0 = A100, 8.6 = A10/3090,
        # 9.0 = H100. +PTX on the last one keeps it forward-compatible with newer cards.
        arch_list = "6.0;6.1;7.0;7.5;8.0;8.6;9.0+PTX"
    elif major == 11:
        arch_list = "5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    else:
        arch_list = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"      # the original list

    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    print(f"[pointnet2_ops] CUDA {cuda_version or 'unknown'} -> "
          f"TORCH_CUDA_ARCH_LIST={arch_list}")
else:
    print(f"[pointnet2_ops] using TORCH_CUDA_ARCH_LIST from the environment: "
          f"{os.environ['TORCH_CUDA_ARCH_LIST']}")
setup(
    name="pointnet2_ops",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
