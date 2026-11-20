#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <string.h>
#include <utility>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

// number of bounce rays written by the last Shade pass
__device__ int d_shade_new_ray_counter = 0;

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}


__global__ void generate_primary_ray(ray *generate_buffer, int max_x, int max_y, camera **cam, curandState *rand_state){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    generate_buffer[pixel_index] = r;
    generate_buffer[pixel_index].pixel_index = pixel_index;
    rand_state[pixel_index] = local_rand_state;
}

// Intersects every ray in the buffer with the scene; stores hit distance and primitive index on the ray.
__global__ void Extend(ray *generate_buffer, int no_rays, hitable **world){
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_index >= no_rays) return;

    ray cur_ray = generate_buffer[thread_index];

    float dist;
    int objIdx;
    bool hit = (*world)->hit_wavefront(cur_ray, 0.001f, FLT_MAX, dist, objIdx);

    if(hit){
        generate_buffer[thread_index].t = dist;
        generate_buffer[thread_index].prim_idx = objIdx;
    }
}

// Shades every hit; scattering paths write their bounce ray to new_ray_buffer.
__global__ void Shade(ray *generate_buffer, ray *new_ray_buffer, vec3 *accumulator, int no_rays,  hitable **world, curandState *rand_state){
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_index >= no_rays) return;
    ray r = generate_buffer[thread_index];

    int primIdx = r.primIdx();
    int pixel_index = r.pixelIndex();
    vec3 Direction = r.direction();

    if (primIdx != -1){
        float Dist = r.distance();
        vec3 I = r.point_at_parameter(Dist);
        vec3 N = (*world)->prim_normal(primIdx, I);

        material* mat_ptr = (*world)->get_mat_ptr(primIdx);

        vec3 color;
        ray bounce_ray;
        curandState local_rand_state = rand_state[pixel_index];

        hit_record rec = {Dist, I, N, mat_ptr};

        if(mat_ptr != nullptr && mat_ptr->scatter(r, rec, color, bounce_ray, &local_rand_state)){
            int ei = atomicAdd(&d_shade_new_ray_counter, 1);
            bounce_ray.pixel_index = pixel_index;
            new_ray_buffer[ei] = bounce_ray;
            accumulator[pixel_index] *= color;
        } else{
            accumulator[pixel_index] = vec3(0.0,0.0,0.0);
        }
        rand_state[pixel_index] = local_rand_state;

    } else{
        vec3 unit_direction = unit_vector(Direction);
        float t = 0.5f*(unit_direction.y() + 1.0f);
        vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
        accumulator[pixel_index] *= c;
    }
}

// Paths still alive after the bounce limit contribute nothing, as in color().
__global__ void terminate_remaining(ray *ray_buffer, vec3 *accumulator, int no_rays){
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_index >= no_rays) return;
    accumulator[ray_buffer[thread_index].pixelIndex()] = vec3(0.0,0.0,0.0);
}


__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void resetCounter() {
    d_shade_new_ray_counter = 0;
}

__global__ void initializeAccumulator(vec3* accumulator, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        accumulator[idx] = vec3(1.0f, 1.0f, 1.0f);
    }
}

__global__ void accumulatorrays(vec3* d_general_accumulator, vec3* accumulator, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int thread_index = j*max_x + i;
    d_general_accumulator[thread_index] += accumulator[thread_index];
}

int main(int argc, char **argv) {

    cudaSetDevice(0);

    // default: wavefront; pass --megakernel for the original renderer
    bool wavefront = !(argc > 1 && strcmp(argv[1], "--megakernel") == 0);

    cudaDeviceProp prop;
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&prop, device);

        std::cerr << "Device " << device << ": " << prop.name << std::endl;
        std::cerr << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cerr << "  Total global memory: " << prop.totalGlobalMem << " bytes" << std::endl;
        std::cerr << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cerr << std::endl;
    }

    int nx = 1920;
    int ny = 1080;
    int ns = 10;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "(" << (wavefront ? "wavefront" : "megakernel") << ").\n";

    int num_pixels = nx*ny;

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    hitable **d_list;
    int num_hitables = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks3D(nx/tx+1,ny/ty+1);
    dim3 threads3D(tx,ty);
    render_init<<<blocks3D, threads3D>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if(wavefront){
        size_t d_general_accumulator_size = num_pixels*sizeof(vec3);
        vec3 *d_general_accumulator;
        checkCudaErrors(cudaMallocManaged((void **)&d_general_accumulator, d_general_accumulator_size));

        // two ray buffers, swapped every bounce
        ray *generate_buffer;
        ray *shade_new_ray_buffer;
        checkCudaErrors(cudaMalloc((void **)&generate_buffer, num_pixels*sizeof(ray)));
        checkCudaErrors(cudaMalloc((void **)&shade_new_ray_buffer, num_pixels*sizeof(ray)));

        vec3 *d_accumulator;
        checkCudaErrors(cudaMalloc((void **)&d_accumulator, num_pixels*sizeof(vec3)));

        const int threads = 64;

        for(int s=0; s < ns; s++) {
            int no_of_rays_to_trace = num_pixels;

            initializeAccumulator<<<(num_pixels + 256 - 1) / 256, 256>>>(d_accumulator, num_pixels);
            checkCudaErrors(cudaGetLastError());

            // Phase 1 (Generate)
            // Generates one primary ray per pixel and stores it in a buffer.
            generate_primary_ray<<<blocks3D, threads3D>>>(generate_buffer, nx, ny, d_camera, d_rand_state);
            checkCudaErrors(cudaGetLastError());

            for(int i = 0; i < 50 && no_of_rays_to_trace > 0; i++) {
                int blocks = (no_of_rays_to_trace + threads - 1) / threads;

                // Phase 2 (Extend)
                // Executed only after the previous phase has completed for all rays.
                // Reads the ray buffer and intersects each ray with the scene.
                // The intersection result is stored on the ray in the same buffer.
                Extend<<<blocks, threads>>>(generate_buffer, no_of_rays_to_trace, d_world);
                checkCudaErrors(cudaGetLastError());

                resetCounter<<<1, 1>>>();
                checkCudaErrors(cudaGetLastError());

                // Phase 3 (Shade)
                // Executed after phase 2 is completely done.
                // Takes the intersection result from phase 2 and evaluates the shading model for each path.
                // This may or may not generate a new ray, depending on whether the path was terminated.
                // A path that spawns a new ray (the path is 'extended') writes it to the new ray buffer.
                // Shadow rays for next event estimation are not used: Ray Tracing in One Weekend has no explicit light sampling.
                Shade<<<blocks, threads>>>(generate_buffer, shade_new_ray_buffer, d_accumulator, no_of_rays_to_trace, d_world, d_rand_state);
                checkCudaErrors(cudaGetLastError());

                // blocking copy: the host needs the count to size the next launch
                checkCudaErrors(cudaMemcpyFromSymbol(&no_of_rays_to_trace, d_shade_new_ray_counter, sizeof(int), 0, cudaMemcpyDeviceToHost));
                std::swap(generate_buffer, shade_new_ray_buffer);
            }

            if(no_of_rays_to_trace > 0){
                terminate_remaining<<<(no_of_rays_to_trace + threads - 1) / threads, threads>>>(generate_buffer, d_accumulator, no_of_rays_to_trace);
                checkCudaErrors(cudaGetLastError());
            }

            accumulatorrays<<<blocks3D, threads3D>>>(d_general_accumulator,  d_accumulator, nx, ny);
            checkCudaErrors(cudaGetLastError());
        }
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(generate_buffer));
        checkCudaErrors(cudaFree(shade_new_ray_buffer));
        checkCudaErrors(cudaFree(d_accumulator));

        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds.\n";


        std::cout << "P3\n" << nx << " " << ny << "\n255\n";
        for (int j = ny-1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j*nx + i;
                int ir = int(255.99*sqrt(d_general_accumulator[pixel_index].r()/float(ns)));
                int ig = int(255.99*sqrt(d_general_accumulator[pixel_index].g()/float(ns)));
                int ib = int(255.99*sqrt(d_general_accumulator[pixel_index].b()/float(ns)));
                std::cout << ir << " " << ig << " " << ib << "\n";
            }
        }

        checkCudaErrors(cudaFree(d_general_accumulator));
    }
    else
    {
        size_t fb_size = num_pixels*sizeof(vec3);
        vec3 *fb;
        checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

        render<<<blocks3D, threads3D>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds.\n";

        // Output FB as Image
        std::cout << "P3\n" << nx << " " << ny << "\n255\n";
        for (int j = ny-1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j*nx + i;
                int ir = int(255.99*fb[pixel_index].r());
                int ig = int(255.99*fb[pixel_index].g());
                int ib = int(255.99*fb[pixel_index].b());
                std::cout << ir << " " << ig << " " << ib << "\n";
            }
        }

        checkCudaErrors(cudaFree(fb));
    }

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    cudaDeviceReset();
}
