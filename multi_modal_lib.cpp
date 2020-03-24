



#include "multi_modal.hpp"
#include <vector>
#include <sstream>

#include "multi_modal_lib.h"

struct multi_modal_wrapper {
    multi_modal<std::vector<float>> ds;
    unsigned long dimensions;
};

multi_modal_wrapper * mm_create(unsigned long dimensions, unsigned long maximum_nodes) {
    return new multi_modal_wrapper{multi_modal<std::vector<float>>(maximum_nodes), dimensions};
}

void mm_destroy(multi_modal_wrapper * wrapper) {
    delete wrapper;
}


unsigned long mm_get_dimensions(multi_modal_wrapper * wrapper) {
    if (wrapper == nullptr) return 0;
    return wrapper->dimensions;
}

void mm_insert(multi_modal_wrapper * wrapper, float * sample, unsigned long dimensions) {
    wrapper->ds.insert(std::vector<float>(sample, sample + dimensions));
}

void mm_find_peak(
    multi_modal_wrapper * wrapper, 
    float * sample, unsigned long dimensions, 
    distribution_wrapper ** wrappers, unsigned long * wrapper_count) 
{
    auto peak = wrapper->ds.find_peak(std::vector<float>(sample, sample + dimensions));
    
    *wrappers = new distribution_wrapper[1];
    *wrapper_count = 1;

    (*wrappers)[0].mean = new float[wrapper->dimensions];
    std::copy(peak.second.mean.begin(), peak.second.mean.end(), (*wrappers)[0].mean);
    (*wrappers)[0].mean_size = wrapper->dimensions;
    (*wrappers)[0].standard_deviation = peak.second.standard_deviation();
    (*wrappers)[0].sample_count = peak.second.count;
    (*wrappers)[0].id = peak.first;
}

unsigned long mm_get_count(multi_modal_wrapper * wrapper) {
    return wrapper->ds.get_count();
}
void mm_extract_peaks(multi_modal_wrapper * wrapper, distribution_wrapper ** wrappers, unsigned long * wrapper_count) {
    auto peaks = wrapper->ds.extract_peaks();

    *wrappers = new distribution_wrapper[peaks.size()];
    *wrapper_count = peaks.size();

    for(unsigned long i = 0; i < peaks.size(); i++) {
        (*wrappers)[i].mean = new float[wrapper->dimensions];
        std::copy(peaks[i].second.mean.begin(), peaks[i].second.mean.end(), (*wrappers)[i].mean);
        (*wrappers)[i].mean_size = wrapper->dimensions;
        (*wrappers)[i].standard_deviation = peaks[i].second.standard_deviation();
        (*wrappers)[i].sample_count = peaks[i].second.count;
        (*wrappers)[i].id = peaks[i].first;
    }
}
void mm_destroy_peaks(multi_modal_wrapper * wrapper, distribution_wrapper * wrappers, unsigned long wrapper_count) {
    for(unsigned long i = 0; i < wrapper_count; i++) {
        delete [] wrappers[i].mean;
    }
    delete [] wrappers;
}

void mm_serialize(multi_modal_wrapper * wrapper, char ** output_buf, unsigned long * output_size) {
    std::stringstream ss;
    wrapper->ds.serialize(ss);

    auto s = ss.str();

    *output_size = (unsigned long)s.length();

    *output_buf = new char[s.length()];

    std::copy(s.begin(), s.end(), *output_buf);

}
void mm_destroy_serialize_buffer(multi_modal_wrapper * wrapper, char * output_buf, unsigned long output_size) {
    delete [] output_buf;
}
void mm_deserialize(multi_modal_wrapper * wrapper, char * input_buf, unsigned long input_size) {
    std::stringstream ss(std::string(input_buf, input_size));

    wrapper->ds.deserialize(ss);
}