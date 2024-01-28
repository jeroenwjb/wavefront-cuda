#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere: public hitable  {
    public:
        __device__ sphere() {}
        __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m)  {};
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        __device__ virtual bool hit_wavefront(const ray& r, float tmin, float tmax, float& dist, int& objIdx) const;
        __device__ virtual vec3 prim_normal(int primIdx, vec3 I);
        __device__ virtual hitable* get_hittable(int primIdx);
        __device__ virtual material* get_mat_ptr(int primIdx);

        vec3 center;
        float radius;
        material *mat_ptr;
};

__device__ bool sphere::hit_wavefront(const ray& r, float t_min, float t_max, float& dist, int& objIdx) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            dist = temp;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            dist = temp;
            return true;
        }
    }
    return false;
}

__device__ vec3 sphere::prim_normal(int primIdx, vec3 I){
    return  (I - center) / radius;
}

__device__ hitable* sphere::get_hittable(int primIdx){
    printf("This code should not run");
    return nullptr;
}

__device__ material* sphere::get_mat_ptr(int primIdx){
    // printf("This code should not run");
    return mat_ptr;
}



__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}
#endif
