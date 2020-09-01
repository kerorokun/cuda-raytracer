#include <math.h>   
#include <stdlib.h> 
#include <stdio.h> 
#include <fstream>
#include <string>
#include <iostream>
#include "common/fmt.hpp"
#include "common/utils.hpp"
#include "curand_kernel.h"

struct Vec { 
    double x, y, z;                  // position, also color (r,g,b)

    __device__ __host__ Vec operator+(const Vec &b) const {
        Vec v;
        v.x = x+b.x;
        v.y = y+b.y;
        v.z = z+b.z;
        return v;
    }
    __device__ __host__ Vec operator-(const Vec &b) const {
        Vec v;
        v.x = x - b.x;
        v.y = y - b.y;
        v.z = z - b.z;
        return v;
    }
    __device__ __host__ Vec operator*(double b) const {
        Vec v;
        v.x = x * b;
        v.y = y * b;
        v.z = z * b;
        return v;
    }
    __device__ __host__ Vec operator%(Vec&b){
        Vec v;
        v.x = y * b.z - z * b.y;
        v.y = z * b.x - x * b.z;
        v.z = x * b.y - y * b.x;
        return v;
    }

    __device__ __host__ Vec mult(const Vec &b) const {
        Vec v;
        v.x = x * b.x;
        v.y = y * b.y;
        v.z = z * b.z;
        return v;
    }
    __device__ __host__ Vec& norm() { return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
    __device__ __host__ double dot(const Vec &b) const { return x*b.x+y*b.y+z*b.z; } // cross:
};


struct Ray { Vec o, d; };
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct Sphere {
	double rad;       // radius
	Vec p, e, c;      // position, emission, color
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)

	__device__ __host__ double intersect(const Ray &r) const { // returns distance, 0 if nohit
		Vec op = p-r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		double t, eps=1e-4, b=op.dot(r.d), det=b*b-op.dot(op)+rad*rad;
		if (det<0) return 0; else det=sqrt(det);
		return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
	}
};


__device__ __host__ Vec new_vec(double x_=0, double y_=0, double z_=0) {
    Vec v;
    v.x = x_;
    v.y = y_;
    v.z = z_;
    return v;
}

__device__ __host__ Ray new_ray(Vec o_, Vec d_) {
    Ray r;
    r.o = o_;
    r.d = d_;
    return r;
}

__device__ __host__ Sphere new_sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) {
    Sphere s;
    s.rad  = rad_;
    s.p    = p_;
    s.e    = e_;
    s.c    = c_;
    s.refl = refl_;
    return s;
}

// CUDA FUNCTIONS ===========================================================
#define MAX_SPHERES 9
#define BLOCK_SIZE 64
#define NUM_CURAND 196608

static __constant__ Sphere SPHERES[MAX_SPHERES];

__device__ __host__ inline double clamp(double x) {
    return x<0 ? 0 : x>1 ? 1 : x;
}

int toInt(double x) {
    return int(pow(clamp(x),1/2.2)*255+.5);
}

__device__ bool intersect(const Ray &r, double &t, int &id, int num_spheres) {
    double d;
    double inf = t = 1e20;
    for(int i = num_spheres; i--;)
    	if( (d = SPHERES[i].intersect(r)) && d<t ) {
	    t=d;
	    id=i;
	}
    return t < inf;
}

__device__ Vec linear_radiance(const Ray &r_, int depth_, int num_spheres, curandState *Xi){
	double t;                               // distance to intersection
	int id=0;                               // id of intersected object
	Ray r=r_;
	int depth=depth_;
	// L0 = Le0 + f0*(L1)
	//    = Le0 + f0*(Le1 + f1*L2)
	//    = Le0 + f0*(Le1 + f1*(Le2 + f2*(L3))
	//    = Le0 + f0*(Le1 + f1*(Le2 + f2*(Le3 + f3*(L4)))
	//    = ...
	//    = Le0 + f0*Le1 + f0*f1*Le2 + f0*f1*f2*Le3 + f0*f1*f2*f3*Le4 + ...
	//
	// So:
	// F = 1
	// while (1){
	//   L += F*Lei
	//   F *= fi
	// }
	Vec cl = new_vec(0,0,0);   // accumulated color
	Vec cf = new_vec(1,1,1);  // accumulated reflectance
	while (1){
		if (!intersect(r, t, id, num_spheres)) return cl; // if miss, return black
		const Sphere &obj = SPHERES[id];        // the hit object
		Vec x=r.o+r.d*t, n=(x-obj.p).norm(), nl=n.dot(r.d)<0?n:n*-1, f=obj.c;
		double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl
		cl = cl + cf.mult(obj.e);
		if (++depth>5) if (curand_uniform(Xi)<p) f=f*(1/p); else return cl; //R.R.
		cf = cf.mult(f);
		if (obj.refl == DIFF){                  // Ideal DIFFUSE reflection
			double r1=2*M_PI*curand_uniform(Xi), r2=curand_uniform(Xi), r2s=sqrt(r2);
			Vec w=nl, u=((fabs(w.x)>.1? new_vec(0,1):new_vec(1))%w).norm(), v=w%u;
			Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();
			//return obj.e + f.mult(radiance(Ray(x,d),depth,Xi));
			r = new_ray(x,d);
			continue;
		} else if (obj.refl == SPEC){           // Ideal SPECULAR reflection
			//return obj.e + f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,Xi));
			r = new_ray(x,r.d-n*2*n.dot(r.d));
			continue;
		}
		Ray reflRay = new_ray(x, r.d-n*2*n.dot(r.d));     // Ideal dielectric REFRACTION
		bool into = n.dot(nl)>0;                // Ray from outside going in?
		double nc=1, nt=1.5, nnt=into?nc/nt:nt/nc, ddn=r.d.dot(nl), cos2t;
		if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0){    // Total internal reflection
			//return obj.e + f.mult(radiance(reflRay,depth,Xi));
			r = reflRay;
			continue;
		}
		Vec tdir = (r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();
		double a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:tdir.dot(n));
		double Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P);
        
        if (curand_uniform(Xi)<P){
			cf = cf*RP;
			r = reflRay;
		} else {
			cf = cf*TP;
			r = new_ray(x,tdir);
		}
		continue;
	}
}


__global__ void calc_pixel(Vec *out, int samps, int offset,
                           int num_spheres, 
                           int w, int h,
                           Ray cam, Vec cx, Vec cy,
                           curandState __restrict__ *states) {
    // Calculates a single pixel in the final image
    // Returns a color vector that is later written to the final image.
    int t = blockIdx.x * blockDim.x + threadIdx.x + offset;
    curandState state = states[t - offset];

    int y = (((h-1) * w) - t)/w;
    int x = (((h-1) * w) - t)%w;

    if (t < w*h) {

        int i = (h-y-1) * w + x;

        for (int sy = 0; sy < 2; sy++) {     // 2x2 subpixel rows
            for (int sx = 0; sx < 2; sx++) {        // 2x2 subpixel cols
                Vec r = new_vec();

                for (int s = 0; s < samps; s++) {
                    double r1 = 2*curand_uniform(&state), dx=r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1);
                    double r2 = 2*curand_uniform(&state), dy=r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2);

                    Vec d = cx*( ( (sx+.5 + dx)/2 + x)/w - .5) +
                        cy*( ( (sy+.5 + dy)/2 + y)/h - .5) + cam.d;
                    d = d.norm();
                    Vec rad = linear_radiance(new_ray(cam.o+d*140,d),0, num_spheres, &state);

                    r = r + rad * (1./samps);
                } // Camera rays are pushed ^^^^^ forward to start in interior

                out[i] = out[i] + new_vec(clamp(r.x),clamp(r.y),clamp(r.z))*.25;
            }
        }

    }
}

__global__ void init_curand(curandState __restrict__ *states) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(t, t, 0, &states[t]);
}

int main(int argc, char *argv[]) {
    timer_start("Starting program."); //@@ start a timer

    int w=1024, h=768; // # samples
    int samps = argc>=2 ? atoi(argv[1])/4 : 250;

    const int num_streams = 4;
    dim3 grid(ceil((w*h)/(BLOCK_SIZE * num_streams)), 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    curandState *device_states;
    cudaMalloc((void **) &device_states, sizeof(curandState) * NUM_CURAND);

    // DEVICE - Start initializing the curandstate objects in the background
    init_curand<<<grid, block>>>(device_states);

    // HOST - Start initialzing everything else
    Sphere *spheres;// = (Sphere *)malloc(NUM_SPHERES * sizeof(Sphere));
    cudaHostAlloc((void**)&spheres, MAX_SPHERES * sizeof(Sphere), cudaHostAllocDefault);

    std::ifstream f_in(argv[2]);

    int num_spheres = 0;
    for (int i = 0; i < MAX_SPHERES; i++) {
        std::string rad;
        std::string px, py, pz;
        std::string ex, ey, ez;
        std::string cx, cy, cz;
        std::string refl;
        
        if (std::getline(f_in, rad, ',') &&
            std::getline(f_in, px, ',') && std::getline(f_in, py, ',') && std::getline(f_in, pz, ',') &&
            std::getline(f_in, ex, ',') && std::getline(f_in, ey, ',') && std::getline(f_in, ez, ',') &&
            std::getline(f_in, cx, ',') && std::getline(f_in, cy, ',') && std::getline(f_in, cz, ',') &&
            std::getline(f_in, refl)) {

            if (refl.compare("DIFF") == 0) {
                spheres[i] = new_sphere(std::stod(rad),
                                        new_vec(std::stod(px), std::stod(py), std::stod(pz)),
                                        new_vec(std::stod(ex), std::stod(ey), std::stod(ez)),
                                        new_vec(std::stod(cx), std::stod(cy), std::stod(cz)),
                                        DIFF);
            } else if (refl.compare("SPEC") == 0) {
                spheres[i] = new_sphere(std::stod(rad),
                                        new_vec(std::stod(px), std::stod(py), std::stod(pz)),
                                        new_vec(std::stod(ex), std::stod(ey), std::stod(ez)),
                                        new_vec(std::stod(cx), std::stod(cy), std::stod(cz)),
                                        SPEC);
            } else if (refl.compare("REFR") == 0) {
                spheres[i] = new_sphere(std::stod(rad),
                                        new_vec(std::stod(px), std::stod(py), std::stod(pz)),
                                        new_vec(std::stod(ex), std::stod(ey), std::stod(ez)),
                                        new_vec(std::stod(cx), std::stod(cy), std::stod(cz)),
                                        REFR);
            }
            
            num_spheres++;
        } else {
            spheres[i] = new_sphere(0, new_vec(), new_vec(), new_vec(), DIFF);
        }
    }
    

    f_in.close();


    cudaStream_t stream[num_streams];
    for (int i = 0; i < num_streams; ++i)
      cudaStreamCreate(&stream[i]);
    
    Ray cam = new_ray(new_vec(50,52,295.6), new_vec(0,-0.042612,-1).norm()); // cam pos, dir
	Vec cx = new_vec(w*.5135/h), cy = (cx%cam.d).norm()*.5135;

    Vec *host_out = (Vec *)malloc(sizeof(Vec) * w * h);

    Vec *device_out;
    cudaMalloc((void **) &device_out, sizeof(Vec) * w * h);

    int num_elems_per_segment = w * h / num_streams;
    int segment_size = sizeof(Vec) * w * h / num_streams;

    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);

    // DEVICE - Synchronize with CUDA to finish curandStates initialization
    cudaDeviceSynchronize();

    for (int i = 0, offset = 0; i < num_streams; i++, offset += num_elems_per_segment) {
        cudaMemcpyToSymbolAsync(SPHERES, spheres, MAX_SPHERES * sizeof(Sphere), 0, cudaMemcpyHostToDevice, stream[i]);
        calc_pixel<<<grid, block, 0, stream[i]>>>(device_out, samps, offset, num_spheres, w, h, cam, cx, cy, device_states);
    }

    for (int i = 0, offset = 0; i < num_streams; i++, offset += num_elems_per_segment) {
        cudaMemcpyAsync(&host_out[offset], &device_out[offset], segment_size, cudaMemcpyDeviceToHost, stream[i]);

        for (int j=0; j < num_elems_per_segment; j++)
    	   fprintf(f,"%d %d %d ", toInt(host_out[j + offset].x), toInt(host_out[j + offset].y), toInt(host_out[j + offset].z));
    }
    
    fclose(f);

    for (int i = 0; i < num_streams; ++i)
      cudaStreamDestroy(stream[i]);

    cudaFree(device_out);
    cudaFree(device_states);
    free(host_out);
    cudaFreeHost(spheres);

    timer_stop();

    return 0;
}
