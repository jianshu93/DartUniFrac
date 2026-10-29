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

/* data/test.nwk as dartunifrac-arrow's tests/common/mod.rs numbers it. */
#define N_NODES 13
static const int64_t PARENT[N_NODES] = {-1, -1, 1, 2, 2, 4, 4, 1, 7, 7, 9, 9, 9};
static const double LENS[N_NODES] = {
    0.0, 0.0, 0.30000001192092896, 0.10000000149011612, 0.019999999552965164,
    0.05000000074505806, 0.05000000074505806, 0.4000000059604645,
    0.20000000298023224, 0.05000000074505806, 0.10000000149011612,
    0.15000000596046448, 0.20000000298023224,
};

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

static void frees_accept_null(void) {
    df_ctx_free(NULL);
    df_tree_free(NULL);
    CHECK(df_ctx_last_error(NULL) == NULL);
}

int main(void) {
    version();
    contexts();
    trees();
    frees_accept_null();
    puts("dartunifrac C harness: ok");
    return 0;
}
