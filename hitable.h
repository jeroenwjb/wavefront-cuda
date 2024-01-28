#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

class hitable  {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
        __device__ virtual bool hit_wavefront(const ray& r, float t_min, float t_max, float& dist, int& objIdx) const = 0;
        __device__ virtual vec3 prim_normal(int primIdx, vec3 I);
        __device__ virtual hitable* get_hittable(int primIdx);
        __device__ virtual material* get_mat_ptr(int primIdx);
};

#endif
