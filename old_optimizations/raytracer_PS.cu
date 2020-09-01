#include <math.h>
#include <stdlib.h>
#include <stdio.h>
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
        /* return Vec(x-b.x,y-b.y,z-b.z); */
    }
    __device__ __host__ Vec operator*(double b) const {
        Vec v;
        v.x = x * b;
        v.y = y * b;
        v.z = z * b;
        return v;
        /* return Vec(x*b,y*b,z*b); */
    }
    __device__ __host__ Vec operator%(Vec&b){
        Vec v;
        v.x = y * b.z - z * b.y;
        v.y = z * b.x - x * b.z;
        v.z = x * b.y - y * b.x;
        return v;
        /* return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x); */
    }

    __device__ __host__ Vec mult(const Vec &b) const {
        Vec v;
        v.x = x * b.x;
        v.y = y * b.y;
        v.z = z * b.z;
        return v;
        /* return Vec(x*b.x,y*b.y,z*b.z); */
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
#define NUM_SPHERES 9

static __constant__ Sphere SPHERES[NUM_SPHERES];

__device__ __host__ inline double clamp(double x) {
    return x<0 ? 0 : x>1 ? 1 : x;
}

int toInt(double x) {
    return int(pow(clamp(x),1/2.2)*255+.5);
}

__device__ bool intersect(const Ray &r, double &t, int &id) {
    int n = NUM_SPHERES;
    double d;
    double inf = t = 1e20;
    for(int i = int(n); i--;)
    	if( (d = SPHERES[i].intersect(r)) && d<t ) {
	    t=d;
	    id=i;
	}
    return t < inf;
}

#define STACK_SIZE 100

__device__ Vec linear_radiance(const Ray &r_, int depth_, curandState *Xi){
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
		if (!intersect(r, t, id)) return cl; // if miss, return black
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
		// return obj.e + f.mult(curand_uniform(Xi)<P ?
		//                       radiance(reflRay,    depth,Xi)*RP:
		//                       radiance(Ray(x,tdir),depth,Xi)*TP);
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


__global__ void calc_pixel(Vec *out, int samps) {
    // Calculates a single pixel in the final image
    // Returns a color vector that is later written to the final image.
    curandState state;
    
	int w=1024, h=768;
    Ray cam = new_ray(new_vec(50,52,295.6), new_vec(0,-0.042612,-1).norm()); // cam pos, dir
	Vec cx = new_vec(w*.5135/h), cy = (cx%cam.d).norm()*.5135;
    
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(t, t, 0, &state);
    for(int idx = 0; idx < 4; idx++){
        int y = t/w + (idx*(h/4));
        int x = t%w;

        int i = (h-y-1) * w + x;
        for (int sy = 0; sy < 2; sy++) {     // 2x2 subpixel rows
            for (int sx = 0; sx < 2; sx++) {        // 2x2 subpixel cols
                Vec r = new_vec();

                for (int s = 0; s < samps; s++) {
                    double r1 = 2*curand_uniform(&state), dx=r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1);
                    double r2 = 2*curand_uniform(&state), dy=r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2);

                    Vec d = cx*( ( (sx+.5 + dx)/2 + x)/w - .5) +
                            cy*( ( (sy+.5 + dy)/2 + y)/h - .5) + cam.d;

                    r = r + linear_radiance(new_ray(cam.o+d.norm()*140,d.norm()),0, &state) * (1./samps);
                } // Camera rays are pushed ^^^^^ forward to start in interior

                out[i] = out[i] + new_vec(clamp(r.x),clamp(r.y),clamp(r.z))*.25;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    float BLOCK_SIZE = 512;

    timer_start("Starting program."); //@@ start a timer
    
    Sphere *spheres = (Sphere *)malloc(NUM_SPHERES * sizeof(Sphere));
	spheres[0] = new_sphere(1e5,  new_vec( 1e5+1,40.8,81.6),  new_vec(),new_vec(.75,.25,.25),DIFF);//Left
	spheres[1] = new_sphere(1e5,  new_vec(-1e5+99,40.8,81.6), new_vec(),new_vec(.25,.25,.75),DIFF);//Rght
	spheres[2] = new_sphere(1e5,  new_vec(50,40.8, 1e5),      new_vec(),new_vec(.75,.75,.75),DIFF);//Back
	spheres[3] = new_sphere(1e5,  new_vec(50,40.8,-1e5+170),  new_vec(),new_vec(),           DIFF);//Frnt
	spheres[4] = new_sphere(1e5,  new_vec(50, 1e5, 81.6),     new_vec(),new_vec(.75,.75,.75),DIFF);//Botm
	spheres[5] = new_sphere(1e5,  new_vec(50,-1e5+81.6,81.6), new_vec(),new_vec(.75,.75,.75),DIFF);//Top
	spheres[6] = new_sphere(16.5, new_vec(27,16.5,47),        new_vec(),new_vec(1,1,1)*.999, SPEC);//Mirr
	spheres[7] = new_sphere(16.5, new_vec(73,16.5,78),        new_vec(),new_vec(1,1,1)*.999, REFR);//Glas
	spheres[8] = new_sphere(600,  new_vec(50,681.6-.27,81.6), new_vec(12,12,12),  new_vec(), DIFF);//Lite

    // Copy the spheres to constant memory
    cudaMemcpyToSymbol(SPHERES, spheres, NUM_SPHERES * sizeof(Sphere));

    int w=1024, h=768; // # samples
    int samps = argc==2 ? atoi(argv[1])/4 : 250;
    Vec *host_out = (Vec *)malloc(sizeof(Vec) * w * h);

    Vec *device_out;
    cudaMalloc((void **) &device_out, sizeof(Vec) * w * h);

    printf("This is Chris's 1-D optimization.\n");

    printf("Render starting!\nBlock size is %i\n", int(BLOCK_SIZE));
    dim3 grid(ceil((w*h)/(4*BLOCK_SIZE)), 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    calc_pixel<<<grid, block>>>(device_out, samps);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaMemcpy(host_out, device_out, sizeof(Vec) * w * h, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    cudaFree(device_out);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Image rendered in %f milliseconds!\n", milliseconds);

    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i=0; i<w*h; i++)
	fprintf(f,"%d %d %d ", toInt(host_out[i].x), toInt(host_out[i].y), toInt(host_out[i].z));
    fclose(f);

    free(host_out);
    free(spheres);

    timer_stop();
    
    return 0;
}
