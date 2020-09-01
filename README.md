# CUDA-raytracer

This is a CUDA implementation of Kevin Beason's raytracer (https://www.kevinbeason.com/smallpt/). 
<img src="https://github.com/kerorokun/cuda-raytracer/blob/master/images/result_25k.png" width="500" />

This was both an exercise in how to convert a sequential CPU code into a parallelized GPU code as well as an exercise in GPU optimizations. The final GPU program works like the following:

<img src="https://github.com/kerorokun/cuda-raytracer/blob/master/images/conversion.png" width="500" />

The optimizations were tested on a GPU server that contained two NVIDIA GTX Titans. While not the most practical testing situation (only one machine was really tested), the project was designed as a learning experiment and so served more to satisfy personal curiosity. The following optimizations were attempted:
* Separate kernel for cuRAND initialization
* Usage of `__restrict__` compiler directive
* Streaming
* Parallelization of subpixels instead of pixels
* Usage of constant and shared memory to store object data

After all the optimizations were implemented the following runtimes are observed (small samples were chosen simply because of how long the CPU algorithm ran in):
Type | 128 Samples | 256 Samples | 512 Samples
------------ | ------------- | ------------- | ------------- |
Sequential | 586,453 ms | 1,288,482 ms | 2,656,404 ms
CUDA (with 8 streams) | 1849 ms | 2124 ms | 2384 ms

<img src="https://github.com/kerorokun/cuda-raytracer/blob/master/images/samples_vs_runtime_comparison.png" width="500" />

Some takeaways:
* Surprisingly the use of `__restrict__` provided noticeable speedups, even though it was only really used for 1 kernel function, showing once more how the use of compiler directives can provide genuine speedup
* The largest bottleneck in the program is the actual initialization of the cuRAND objects. The largest optimization seen was the correction of the cuRAND initialization to use a smaller initial offset in the seeding. None of the other optimizations provided nearly the same kind of speedup. 
* As expected the program is memory bound. It would be nice to somehow perform more work for the memory used, but that's not really a thing at lower samples
