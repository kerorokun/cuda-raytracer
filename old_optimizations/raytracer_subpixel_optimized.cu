#include <math.h> 
#include <stdlib.h> 
#include <stdio.h> 
#include <fstream>
#include <string>
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

#define BLOCK_SIZE 32

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


__global__ void calc_pixel(Vec *out, int samps, int in_offset) {
    // Calculates a single subpixel in the final image
    // Returns a color vector that is later written to the final image.
    __shared__ Vec temp_res[BLOCK_SIZE*4];
    __shared__ curandState states[BLOCK_SIZE];

    int w=1024, h=768;
    Ray cam = new_ray(new_vec(50,52,295.6), new_vec(0,-0.042612,-1).norm()); // cam pos, dir
	Vec cx = new_vec(w*.5135/h), cy = (cx%cam.d).norm()*.5135;

    // Idxing variables
    int full_idx  = blockIdx.x * blockDim.x + threadIdx.x + in_offset;
    int pixel_idx = full_idx / 4;
    int mod_4     = full_idx % 4;
    int sy        = mod_4 / 2;
    int sx        = mod_4 % 2;
    int y = (((h-1) * w) - pixel_idx)/w;
    int x = (((h-1) * w) - pixel_idx)%w;

    // Variables for the (threadIdx.x < BLOCK_SIZE) conversions
    int block_pixel_start = blockDim.x * blockIdx.x / 4;
    int out_offset   = threadIdx.x + (block_pixel_start) % w;
    int pixel_offset = threadIdx.x + block_pixel_start;
    int state_idx = threadIdx.x / 4;
    int out_idx = (h-y-1) * w + (w - 1 - out_offset);

    // Output vector
    Vec r = new_vec();

    if (threadIdx.x < BLOCK_SIZE) {
        curand_init(pixel_offset, 0, 0, &states[threadIdx.x]);
    }
    __syncthreads();

    for (int s = 0; s < samps; s++) {
        double r1 = 2*curand_uniform(&states[state_idx]), dx=r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1);
        double r2 = 2*curand_uniform(&states[state_idx]), dy=r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2);

        Vec d = cx * (((sx+.5 + dx)/2 + x)/w - .5) + cy*(((sy+.5 + dy)/2 + y)/h - .5) + cam.d;
        d = d.norm();
        Vec res = linear_radiance(new_ray(cam.o+d*140,d),0, &states[state_idx]);

        r = r + res * (1./samps);
    } // Camera rays are pushed ^^^^^ forward to start in interior

    temp_res[threadIdx.x] = new_vec(clamp(r.x),clamp(r.y),clamp(r.z))*.25;

    __syncthreads();

    if (threadIdx.x < BLOCK_SIZE) {
        out[out_idx] = temp_res[threadIdx.x*4] + temp_res[threadIdx.x*4+1] + temp_res[threadIdx.x*4+2] + temp_res[threadIdx.x*4+3];
    }
}

int main(int argc, char *argv[]) {
    timer_start("Starting program."); //@@ start a timer

    Sphere *spheres;
    cudaHostAlloc((void**)&spheres, NUM_SPHERES * sizeof(Sphere), cudaHostAllocDefault);

    std::ifstream f_in(argv[2]);
    std::string line;
    char* token;
    int i = 0;
    int j = 0;

    double doubles[10];
    char str[4];

    while (getline(f_in, line))
    {
      token = strtok(&line[0], ",");

      while (token != NULL)
      {
        if (j < 10)
        {
          doubles[j] = atof(token);
          //printf("%f\n", doubles[j]);
        }
        else
        {
          strcpy(str, token);
          //printf("%s\n", str);
        }

        token = strtok(NULL, ",");
        j++;
      }
      j = 0;

      if (strcmp(str, "DIFF") == 0)
        spheres[i] = new_sphere(doubles[0],  new_vec( doubles[1],doubles[2],doubles[3]),
            new_vec(doubles[4], doubles[5], doubles[6]),new_vec(doubles[7],doubles[8],doubles[9]),DIFF);
      else if (strcmp(str, "SPEC") == 0)
        spheres[i] = new_sphere(doubles[0],  new_vec( doubles[1],doubles[2],doubles[3]),
            new_vec(doubles[4], doubles[5], doubles[6]),new_vec(doubles[7],doubles[8],doubles[9]),SPEC);
      else
        spheres[i] = new_sphere(doubles[0],  new_vec( doubles[1],doubles[2],doubles[3]),
            new_vec(doubles[4], doubles[5], doubles[6]),new_vec(doubles[7],doubles[8],doubles[9]),REFR);

      i++;
    }

    f_in.close();

	/*spheres[0] = new_sphere(1e5,  new_vec( 1e5+1,40.8,81.6),  new_vec(),new_vec(.75,.25,.25),DIFF);//Left
	spheres[1] = new_sphere(1e5,  new_vec(-1e5+99,40.8,81.6), new_vec(),new_vec(.25,.25,.75),DIFF);//Rght
	spheres[2] = new_sphere(1e5,  new_vec(50,40.8, 1e5),      new_vec(),new_vec(.75,.75,.75),DIFF);//Back
	spheres[3] = new_sphere(1e5,  new_vec(50,40.8,-1e5+170),  new_vec(),new_vec(),           DIFF);//Frnt
	spheres[4] = new_sphere(1e5,  new_vec(50, 1e5, 81.6),     new_vec(),new_vec(.75,.75,.75),DIFF);//Botm
	spheres[5] = new_sphere(1e5,  new_vec(50,-1e5+81.6,81.6), new_vec(),new_vec(.75,.75,.75),DIFF);//Top
	spheres[6] = new_sphere(16.5, new_vec(27,16.5,47),        new_vec(),new_vec(1,1,1)*.999, SPEC);//Mirr
	spheres[7] = new_sphere(16.5, new_vec(73,16.5,78),        new_vec(),new_vec(1,1,1)*.999, REFR);//Glas
	spheres[8] = new_sphere(600,  new_vec(50,681.6-.27,81.6), new_vec(12,12,12),  new_vec(), DIFF);//Lite
*/
    // Copy the spheres to constant memory
    //cudaMemcpyToSymbol(SPHERES, spheres, NUM_SPHERES * sizeof(Sphere));

    int w=1024, h=768; // # samples
    int samps = argc>=2 ? atoi(argv[1])/4 : 250;

    const int num_streams = 4;
    cudaStream_t stream[num_streams];
    for (int i = 0; i < num_streams; ++i)
      cudaStreamCreate(&stream[i]);

    Vec *host_out = (Vec*)malloc(sizeof(Vec) * w * h);
    //cudaHostAlloc((void**)&host_out, sizeof(Vec) * w * h, cudaHostAllocDefault);

    Vec *device_out;
    cudaMalloc((void **) &device_out, sizeof(Vec) * w * h);

    dim3 grid(ceil((w*h)/(BLOCK_SIZE * num_streams)), 1, 1);
    dim3 block(BLOCK_SIZE*4, 1, 1);

    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);

    int num_elems_per_segment = w * h / num_streams;
    int segment_size = sizeof(Vec) * w * h / num_streams;

    for (int i = 0, offset = 0; i < num_streams; i++, offset += num_elems_per_segment * 4) {
        cudaMemcpyToSymbolAsync(SPHERES, spheres, NUM_SPHERES * sizeof(Sphere), 0, cudaMemcpyHostToDevice, stream[i]);
        calc_pixel<<<grid, block, 0, stream[i]>>>(device_out, samps, offset);
    }

    for (int i = 0, offset = 0; i < num_streams; i++, offset += num_elems_per_segment) {
        cudaMemcpyAsync(&host_out[offset], &device_out[offset], segment_size, cudaMemcpyDeviceToHost, stream[i]);

        for (int j=0; j < num_elems_per_segment; j++) {
            fprintf(f,"%d %d %d ", toInt(host_out[j + offset].x), toInt(host_out[j + offset].y), toInt(host_out[j + offset].z));
        }
    }

    for (int i = 0; i < num_streams; ++i)
      cudaStreamDestroy(stream[i]);

    fclose(f);

    cudaFree(device_out);

    cudaFreeHost(spheres);
    free(host_out);

    timer_stop();

    return 0;
}
