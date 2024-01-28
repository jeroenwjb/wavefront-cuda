#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

class hitable_list: public hitable  {
    public:
        __device__ hitable_list() {}
        __device__ hitable_list(hitable **l, int n) {list = l; list_size = n; }
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        __device__ virtual bool hit_wavefront(const ray& r, float tmin, float tmax, float& dist, int& objIdx) const;
        __device__ virtual vec3 prim_normal(int primIdx, vec3 I);
        __device__ virtual hitable* get_hittable(int primIdx);
        __device__ virtual material* get_mat_ptr(int primIdx);
        hitable **list;
        int list_size;
};

__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
}

__device__ bool hitable_list::hit_wavefront(const ray& r, float t_min, float t_max, float& dist, int& objIdx) const {
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit_wavefront(r, t_min, closest_so_far, dist, objIdx)) {
                hit_anything = true;
                closest_so_far = dist;
                objIdx = i;
            }
        }
        return hit_anything;
}

__device__ vec3 hitable_list::prim_normal(int primIdx, vec3 I) {
    return list[primIdx]->prim_normal(primIdx, I);
}

__device__ hitable* hitable_list::get_hittable(int primIdx) {
    // printf("getting hittable for %i, resulting in: %p", primIdx, list[primIdx]);
    return list[primIdx];
}

__device__ material* hitable_list::get_mat_ptr(int primIdx) {
    // printf("getting hittable for %i, resulting in: %p", primIdx, list[primIdx]);
    return list[primIdx]->get_mat_ptr(primIdx);
}



#endif
