ROOT := ..
LIBDIR := $(ROOT)/target/release

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  FEATURES ?= stdsimd,macos-accelerate
  RPATH := -Wl,-rpath,$(abspath $(LIBDIR))
else ifeq ($(UNAME_S),Linux)
  FEATURES ?= intel-mkl-static,stdsimd
  RPATH := -Wl,-rpath,$(abspath $(LIBDIR))
else
  FEATURES ?= stdsimd
endif

.PHONY: all rust clean

all: example_run

rust:
	cargo build --release --features $(FEATURES)

example_run: example_run.c dartunifrac.h rust
	$(CC) -std=c11 -Wall -Wextra -O2 -I. example_run.c -L$(LIBDIR) -ldartunifrac $(RPATH) -o $@

clean:
	rm -f example_run c_api_unifrac.tsv
