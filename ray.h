#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b) { A = a; B = b;}
        __device__ vec3 origin() const       { return A; }
        __device__ vec3 direction() const    { return B; }
        __device__ vec3 point_at_parameter(float t) const { return A + t*B; }
        __device__ float distance() const { return t; }
        __device__ int pixelIndex() const { return pixel_index; }
        __device__ int primIdx() const { return prim_idx; }

        vec3 A;
        vec3 B;
        float t = 3.0e+20f;
        int prim_idx = -1;
        int pixel_index = -1;

};

#endif
