/* dartunifrac C ABI harness.
 *
 * Proves, from C, what Rust tests cannot: the generated header compiles as
 * C11 with warnings as errors, its struct layouts are the ones the Rust side
 * assumes, the staticlib links, and every handle is freed (run under valgrind
 * by `make leaks`). Behaviour is tested exhaustively in src/tests.rs; this
 * covers each entry point once, the way a C caller holds it.
 *
 * Build and run with `make ctest` (and `make leaks`) from the repo root.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dartunifrac.h"

/* Not assert(): NDEBUG would compile every check away. */
#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,    \
                    #cond);                                                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

/* The layout the Rust side's abi_struct_sizes test asserts too, for every
 * supported target (x86-64, aarch64, wasm32). */
_Static_assert(sizeof(df_status_t) == 4, "df_status_t size");
_Static_assert(sizeof(df_config_t) == 16, "df_config_t size");
_Static_assert(_Alignof(df_config_t) == 8, "df_config_t alignment");
_Static_assert(offsetof(df_config_t, n_threads) == 0, "df_config_t.n_threads");
_Static_assert(offsetof(df_config_t, batch_rows) == 8, "df_config_t.batch_rows");
_Static_assert(sizeof(df_params_t) == 24, "df_params_t size");
_Static_assert(_Alignof(df_params_t) == 8, "df_params_t alignment");
_Static_assert(offsetof(df_params_t, method) == 0, "df_params_t.method");
_Static_assert(offsetof(df_params_t, k) == 4, "df_params_t.k");
_Static_assert(offsetof(df_params_t, seed) == 8, "df_params_t.seed");
_Static_assert(offsetof(df_params_t, bbits) == 16, "df_params_t.bbits");
_Static_assert(offsetof(df_params_t, weighted) == 20, "df_params_t.weighted");
_Static_assert(offsetof(df_params_t, raw_counts) == 21, "df_params_t.raw_counts");
_Static_assert(sizeof(struct ArrowArrayStream) == 5 * sizeof(void *), "ArrowArrayStream size");

/* data/test.nwk as dartunifrac-arrow's tests/common/mod.rs numbers it. */
#define N_NODES 13
static const int64_t PARENT[N_NODES] = {-1, -1, 1, 2, 2, 4, 4, 1, 7, 7, 9, 9, 9};
static const double LENS[N_NODES] = {
    0.0, 0.0, 0.30000001192092896, 0.10000000149011612, 0.019999999552965164,
    0.05000000074505806, 0.05000000074505806, 0.4000000059604645,
    0.20000000298023224, 0.05000000074505806, 0.10000000149011612,
    0.15000000596046448, 0.20000000298023224,
};

/* ---- A C producer of (sample_idx int64, node_idx int64) rows --------------
 *
 * The fixture table's three samples, unweighted, in one batch, written the way
 * any C producer of the Arrow C Stream Interface would. Every struct it hands
 * out is heap-allocated and freed by its release callback, so valgrind sees
 * whether the library releases what it is given exactly once. */

static const int64_t SAMPLES[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
static const int64_t NODES[9] = {3, 6, 8, 5, 10, 11, 3, 6, 12};
static int producer_releases;

typedef struct {
    struct ArrowSchema child[2];
    struct ArrowSchema *children[2];
} schema_owner;

static void release_child_schema(struct ArrowSchema *s) { s->release = NULL; }

static void release_schema(struct ArrowSchema *s) {
    schema_owner *o = s->private_data;
    for (int i = 0; i < 2; i++) {
        if (o->child[i].release != NULL) {
            o->child[i].release(&o->child[i]);
        }
    }
    free(o);
    s->release = NULL;
}

static int producer_get_schema(struct ArrowArrayStream *stream, struct ArrowSchema *out) {
    (void)stream;
    schema_owner *o = calloc(1, sizeof *o);
    const char *names[2] = {"sample_idx", "node_idx"};
    for (int i = 0; i < 2; i++) {
        o->child[i] = (struct ArrowSchema){
            .format = "l", .name = names[i], .release = release_child_schema};
        o->children[i] = &o->child[i];
    }
    *out = (struct ArrowSchema){.format = "+s",
                                .n_children = 2,
                                .children = o->children,
                                .release = release_schema,
                                .private_data = o};
    return 0;
}

typedef struct {
    struct ArrowArray child[2];
    struct ArrowArray *children[2];
    const void *child_buffers[2][2];
    const void *struct_buffers[1];
} batch_owner;

static void release_child_array(struct ArrowArray *a) { a->release = NULL; }

static void release_batch(struct ArrowArray *a) {
    batch_owner *o = a->private_data;
    for (int i = 0; i < 2; i++) {
        if (o->child[i].release != NULL) {
            o->child[i].release(&o->child[i]);
        }
    }
    free(o);
    a->release = NULL;
}

static int producer_get_next(struct ArrowArrayStream *stream, struct ArrowArray *out) {
    int *sent = stream->private_data;
    if (*sent) {
        memset(out, 0, sizeof *out); /* released: the end of the stream */
        return 0;
    }
    *sent = 1;
    batch_owner *o = calloc(1, sizeof *o);
    const int64_t *columns[2] = {SAMPLES, NODES};
    for (int i = 0; i < 2; i++) {
        o->child_buffers[i][0] = NULL; /* no validity bitmap: no nulls */
        o->child_buffers[i][1] = columns[i];
        o->child[i] = (struct ArrowArray){.length = 9,
                                          .n_buffers = 2,
                                          .buffers = o->child_buffers[i],
                                          .release = release_child_array};
        o->children[i] = &o->child[i];
    }
    o->struct_buffers[0] = NULL;
    *out = (struct ArrowArray){.length = 9,
                               .n_buffers = 1,
                               .n_children = 2,
                               .buffers = o->struct_buffers,
                               .children = o->children,
                               .release = release_batch,
                               .private_data = o};
    return 0;
}

static const char *producer_get_last_error(struct ArrowArrayStream *stream) {
    (void)stream;
    return NULL;
}

static void producer_release(struct ArrowArrayStream *stream) {
    free(stream->private_data);
    stream->release = NULL;
    producer_releases++;
}

static void make_producer(struct ArrowArrayStream *stream) {
    *stream = (struct ArrowArrayStream){.get_schema = producer_get_schema,
                                        .get_next = producer_get_next,
                                        .get_last_error = producer_get_last_error,
                                        .release = producer_release,
                                        .private_data = calloc(1, sizeof(int))};
}

static const int64_t *int64_values(const struct ArrowArray *a) {
    return (const int64_t *)a->buffers[1] + a->offset;
}

static void version(void) {
    CHECK(strcmp(df_version(), DF_VERSION) == 0);
}

static void contexts(void) {
    df_ctx_t *ctx = NULL;
    CHECK(df_ctx_new(NULL, &ctx) == DF_OK);
    CHECK(ctx != NULL);
    CHECK(df_ctx_last_error(ctx) == NULL);
    df_ctx_free(ctx);

    df_config_t cfg = {.n_threads = 2, .batch_rows = 0};
    CHECK(df_ctx_new(&cfg, &ctx) == DF_OK);
    df_ctx_free(ctx);

    /* No context exists yet, so the error goes to the thread. */
    cfg.n_threads = -1;
    ctx = (df_ctx_t *)&cfg; /* any non-NULL value: it must come back NULL */
    CHECK(df_ctx_new(&cfg, &ctx) < 0);
    CHECK(ctx == NULL);
    CHECK(df_thread_last_error() != NULL);
    CHECK(strstr(df_thread_last_error(), "n_threads") != NULL);

    CHECK(df_ctx_new(NULL, NULL) == DF_ERR_NULL_POINTER);
}

static void trees(void) {
    df_ctx_t *ctx = NULL;
    CHECK(df_ctx_new(NULL, &ctx) == DF_OK);

    df_tree_t *tree = NULL;
    CHECK(df_tree_new(ctx, PARENT, LENS, N_NODES, &tree) == DF_OK);
    CHECK(tree != NULL);
    df_tree_free(tree);

    int64_t bad[N_NODES];
    memcpy(bad, PARENT, sizeof bad);
    bad[5] = 99;
    CHECK(df_tree_new(ctx, bad, LENS, N_NODES, &tree) == DF_ERR_INVALID_INPUT);
    CHECK(tree == NULL);
    CHECK(df_ctx_last_error(ctx) != NULL);
    CHECK(strstr(df_ctx_last_error(ctx), "99") != NULL);

    CHECK(df_tree_new(ctx, PARENT, LENS, N_NODES, NULL) == DF_ERR_NULL_POINTER);
    df_ctx_free(ctx);
}

/* Sketch, list the samples, stream the distances. Everything that made the
 * stream is freed before it is read: it must not depend on any of it. */
static void sketch_and_distances(int32_t n_threads) {
    df_config_t cfg = {.n_threads = n_threads, .batch_rows = 0};
    df_ctx_t *ctx = NULL;
    CHECK(df_ctx_new(&cfg, &ctx) == DF_OK);
    df_tree_t *tree = NULL;
    CHECK(df_tree_new(ctx, PARENT, LENS, N_NODES, &tree) == DF_OK);

    df_params_t params = {.method = DF_METHOD_DMH,
                          .k = 64,
                          .seed = 1337,
                          .bbits = 16,
                          .weighted = false,
                          .raw_counts = false};
    struct ArrowArrayStream input;
    make_producer(&input);
    producer_releases = 0;
    df_sketches_t *sketches = NULL;
    CHECK(df_sketch(ctx, tree, &params, &input, 3, NULL, &sketches) == DF_OK);
    CHECK(sketches != NULL);
    CHECK(input.release == NULL);
    CHECK(producer_releases == 1);

    struct ArrowArray ids;
    struct ArrowSchema ids_schema;
    CHECK(df_sketch_samples(sketches, &ids, &ids_schema) == DF_OK);
    CHECK(strcmp(ids_schema.format, "l") == 0);
    CHECK(ids.length == 3);
    const int64_t *id = int64_values(&ids);
    CHECK(id[0] == 0 && id[1] == 1 && id[2] == 2);
    ids.release(&ids);
    ids_schema.release(&ids_schema);

    struct ArrowArrayStream out;
    CHECK(df_distances(ctx, sketches, &out) == DF_OK);
    df_sketches_free(sketches);
    df_tree_free(tree);
    df_ctx_free(ctx);

    struct ArrowSchema schema;
    CHECK(out.get_schema(&out, &schema) == 0);
    CHECK(strcmp(schema.format, "+s") == 0 && schema.n_children == 3);
    schema.release(&schema);

    int64_t rows = 0;
    for (;;) {
        struct ArrowArray batch;
        CHECK(out.get_next(&out, &batch) == 0);
        if (batch.release == NULL) {
            break; /* the end */
        }
        CHECK(batch.n_children == 3);
        const int64_t *i = int64_values(batch.children[0]);
        const int64_t *j = int64_values(batch.children[1]);
        const float *d = (const float *)batch.children[2]->buffers[1] + batch.children[2]->offset;
        for (int64_t r = 0; r < batch.length; r++) {
            CHECK(i[r] < j[r]);
            CHECK(d[r] >= 0.0f && d[r] <= 1.0f);
        }
        rows += batch.length;
        batch.release(&batch);
    }
    CHECK(rows == 3); /* 3 samples, 3 pairs */
    out.release(&out);
}

/* A rejected call still consumes its input stream, exactly once. */
static void bad_params_still_release_the_input(void) {
    df_ctx_t *ctx = NULL;
    CHECK(df_ctx_new(NULL, &ctx) == DF_OK);
    df_tree_t *tree = NULL;
    CHECK(df_tree_new(ctx, PARENT, LENS, N_NODES, &tree) == DF_OK);

    df_params_t params = {.method = DF_METHOD_DMH, .k = 0, .seed = 1337, .bbits = 16};
    struct ArrowArrayStream input;
    make_producer(&input);
    producer_releases = 0;
    df_sketches_t *sketches = (df_sketches_t *)&params; /* must come back NULL */
    CHECK(df_sketch(ctx, tree, &params, &input, 3, NULL, &sketches) == DF_ERR_INVALID_INPUT);
    CHECK(sketches == NULL);
    CHECK(input.release == NULL);
    CHECK(producer_releases == 1);
    CHECK(df_ctx_last_error(ctx) != NULL);

    df_tree_free(tree);
    df_ctx_free(ctx);
}

static void frees_accept_null(void) {
    df_ctx_free(NULL);
    df_tree_free(NULL);
    df_sketches_free(NULL);
    CHECK(df_ctx_last_error(NULL) == NULL);
}

int main(void) {
    version();
    contexts();
    trees();
    sketch_and_distances(2); /* a dedicated pool */
    sketch_and_distances(0); /* rayon's global pool */
    bad_params_still_release_the_input();
    frees_accept_null();
    puts("dartunifrac C harness: ok");
    return 0;
}
