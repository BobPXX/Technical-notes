# Technical-notes
#### "error in ms_deformable_im2col_cuda: no kernel image is available for execution on the device" when applying [deformable attention](https://github.com/fundamentalvision/Deformable-DETR).
Solution: When the GPU architecture is too new or old, this error happens. We need to specify the GPU architecture in `setup.py`. For example, for my 3080Ti, I add `"-arch=sm_86","-gencode=arch=compute_86,code=sm_86"` to `extra_compile_args["nvcc"]` 
```
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-arch=sm_86",
            "-gencode=arch=compute_86,code=sm_86", #for too new GPU architecture (e.g., 3080Ti), we need to mannually specify the target GPU architecture
        ]

```
