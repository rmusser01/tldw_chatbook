# build_toolchain.py verifies this existing local tag against the exact supplied image ID before building.
FROM tldw-task32153-onnx-arm64:20260909
LABEL tldw.validation.task="32164" tldw.validation.scope="model-free-native-transport-sanitizers"
RUN timeout 120 apt-get update \
    && DEBIAN_FRONTEND=noninteractive timeout 180 apt-get install -y --no-install-recommends build-essential libasan8 libubsan1 \
    && rm -rf /var/lib/apt/lists/*
