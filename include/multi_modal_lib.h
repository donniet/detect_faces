#ifndef __MULTI_MODAL_LIB_H__
#define __MULTI_MODAL_LIB_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct multi_modal_wrapper multi_modal_wrapper;

typedef struct {
    float * mean;
    unsigned long mean_size;
    float standard_deviation;
    unsigned long sample_count;
    unsigned long id;
} distribution_wrapper;

multi_modal_wrapper * mm_create(unsigned long dimensions, unsigned long maximum_nodes);
void mm_destroy(multi_modal_wrapper * wrapper);

unsigned long mm_get_dimensions(multi_modal_wrapper * wrapper);

void mm_insert(multi_modal_wrapper * wrapper, float * sample, unsigned long dimensions);
unsigned long mm_get_count(multi_modal_wrapper * wrapper);
void mm_extract_peaks(multi_modal_wrapper * wrapper, distribution_wrapper ** wrappers, unsigned long * wrapper_count);
void mm_destroy_peaks(multi_modal_wrapper * wrapper, distribution_wrapper * wrappers, unsigned long wrapper_count);    
void mm_find_peak(
        multi_modal_wrapper * wrapper, 
        float * sample, unsigned long dimensions, 
        distribution_wrapper ** wrappers, unsigned long * wrapper_count);

void mm_serialize(multi_modal_wrapper * wrapper, char ** output_buf, unsigned long * output_size);
void mm_destroy_serialize_buffer(multi_modal_wrapper * wrapper, char * output_buf, unsigned long output_size);
void mm_deserialize(multi_modal_wrapper * wrapper, char * input_buf, unsigned long input_size);

#ifdef __cplusplus
}
#endif

#endif