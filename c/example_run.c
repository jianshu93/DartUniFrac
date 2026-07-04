#include "dartunifrac.h"

#include <stdio.h>

static int print_error(int code) {
    const char *message = dartunifrac_last_error_message();
    fprintf(stderr, "DartUniFrac failed (%d: %s): %s\n",
            code,
            dartunifrac_status_message(code),
            message ? message : "no error message");
    return code;
}

int main(int argc, char **argv) {
    const char *tree = argc > 1 ? argv[1] : "../data/ASVs_aligned.tre";
    const char *biom = argc > 2 ? argv[2] : "../data/ASVs_counts.biom";
    const char *output = argc > 3 ? argv[3] : "c_api_unifrac.tsv";

    DartUniFracConfig config = dartunifrac_config_default();
    config.tree_path = tree;
    config.biom_path = biom;
    config.output_path = output;
    config.seed = 1337;
    config.sketch_size = 2048;
    config.method = "dmh";
    config.bbits = 16;

    int code = dartunifrac_run(&config);
    if (code != DARTUNIFRAC_OK) {
        return print_error(code);
    }
    printf("Wrote %s\n", output);

    DartUniFracMatrix *matrix = NULL;
    code = dartunifrac_compute_matrix(&config, &matrix);
    if (code != DARTUNIFRAC_OK) {
        return print_error(code);
    }

    size_t n = dartunifrac_matrix_sample_count(matrix);
    const float *distances = dartunifrac_matrix_distances(matrix);
    printf("Loaded in-memory matrix with %zu samples\n", n);
    if (n >= 2 && distances) {
        printf("%s vs %s = %.8g\n",
               dartunifrac_matrix_sample_name(matrix, 0),
               dartunifrac_matrix_sample_name(matrix, 1),
               distances[1]);
    }

    dartunifrac_free_matrix(matrix);
    return 0;
}
