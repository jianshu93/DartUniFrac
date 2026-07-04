#ifndef DARTUNIFRAC_H
#define DARTUNIFRAC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DARTUNIFRAC_OK 0
#define DARTUNIFRAC_ERROR -1
#define DARTUNIFRAC_NULL_POINTER -2

typedef struct DartUniFracMatrix DartUniFracMatrix;

typedef struct DartUniFracConfig {
    const char *tree_path;
    const char *input_tsv_path;
    const char *biom_path;
    const char *output_path;
    const char *method;
    size_t sketch_size;
    uint64_t ers_length;
    uint64_t seed;
    uint8_t bbits;
    uint8_t weighted;
    uint8_t raw_counts;
    uint8_t succ;
    uint8_t compress;
    uint8_t pcoa;
    uint8_t streaming;
    size_t block_rows;
    size_t threads;
} DartUniFracConfig;

DartUniFracConfig dartunifrac_config_default(void);
const char *dartunifrac_version(void);
const char *dartunifrac_last_error_message(void);
const char *dartunifrac_status_message(int32_t code);

int32_t dartunifrac_run(const DartUniFracConfig *config);
int32_t dartunifrac_compute_matrix(
    const DartUniFracConfig *config,
    DartUniFracMatrix **out_matrix
);

size_t dartunifrac_matrix_sample_count(const DartUniFracMatrix *matrix);
const char *dartunifrac_matrix_sample_name(
    const DartUniFracMatrix *matrix,
    size_t index
);
const char *const *dartunifrac_matrix_sample_names(
    const DartUniFracMatrix *matrix
);
const float *dartunifrac_matrix_distances(const DartUniFracMatrix *matrix);
void dartunifrac_free_matrix(DartUniFracMatrix *matrix);

#ifdef __cplusplus
}
#endif

#endif
