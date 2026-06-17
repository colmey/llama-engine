# syntax=docker/dockerfile:1
#
# Portable inference image for the agent core:
#   - llama.cpp built from source for Blackwell (sm_120)
#   - llama-swap multi-model proxy (one OpenAI endpoint, hot-swaps GGUFs)
# Models are MOUNTED at runtime (-v ./models:/models), never baked into the image.
#
# Host only needs: NVIDIA driver (>=580) + NVIDIA Container Toolkit. CUDA lives in the image.
#
# Known-good base tags (override with --build-arg if a patch tag is missing):
#   CUDA_DEVEL=13.0.1-devel-ubuntu24.04   CUDA_RUNTIME=13.0.1-runtime-ubuntu24.04
ARG CUDA_DEVEL=13.0.1-devel-ubuntu24.04
ARG CUDA_RUNTIME=13.0.1-runtime-ubuntu24.04

# ---------------------------------------------------------------- build stage
FROM nvidia/cuda:${CUDA_DEVEL} AS build
ARG LLAMA_REF=master
# CPU arch list: 120=Blackwell (this GPU). Add e.g. "89;120" for Ada+Blackwell portability.
ARG CUDA_ARCH=120
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch ${LLAMA_REF} https://github.com/ggml-org/llama.cpp.git .
# NATIVE=OFF + explicit AVX2/FMA/F16C → portable across modern x86-64 (no -march=native lock-in).
# --allow-shlib-undefined: the devel image has no GPU driver, so libggml-cuda.so's libcuda
# (driver API) symbols are left unresolved at build time and satisfied at runtime by the
# NVIDIA Container Toolkit. This mirrors llama.cpp's official cuda.Dockerfile.
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
      -DLLAMA_CURL=ON -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build -j --config Release
# Binaries (llama-server, ...) and their .so deps all land in build/bin — copy that wholesale.

# -------------------------------------------------------------- runtime stage
FROM nvidia/cuda:${CUDA_RUNTIME} AS runtime
ARG LLAMA_SWAP_VER=224
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl libgomp1 libcurl4 gettext-base \
    && rm -rf /var/lib/apt/lists/*
# llama.cpp binaries (llama-server, etc.) + co-located .so deps
COPY --from=build /src/build/bin /opt/llama/bin
ENV PATH=/opt/llama/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/llama/bin:/usr/local/cuda/lib64
# llama-swap proxy
RUN curl -fsSL "https://github.com/mostlygeek/llama-swap/releases/download/v${LLAMA_SWAP_VER}/llama-swap_${LLAMA_SWAP_VER}_linux_amd64.tar.gz" -o /tmp/ls.tgz \
    && tar -xzf /tmp/ls.tgz -C /usr/local/bin llama-swap \
    && rm /tmp/ls.tgz && chmod +x /usr/local/bin/llama-swap
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
