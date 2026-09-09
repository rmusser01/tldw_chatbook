ARG PYTHON_BASE
FROM ${PYTHON_BASE}

LABEL org.opencontainers.image.title="TASK-32153 Linux ARM ONNX qualification" \
      tldw.validation.task="32153" \
      tldw.validation.scope="headless-inference-only"

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates espeak-ng libespeak-ng1 ffmpeg libsndfile1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.in /opt/validation/requirements.in
RUN python -m pip install --index-url https://pypi.org/simple -r /opt/validation/requirements.in \
    && python -m pip check \
    && python -m pip freeze --all > /opt/validation/pip-freeze.txt \
    && dpkg-query -W > /opt/validation/dpkg-packages.txt \
    && python -c "import importlib.util; assert importlib.util.find_spec('torch') is None; import kokoro_onnx, onnxruntime, soundfile; print('Import-only check; no model/session initialized')"

WORKDIR /workspace
CMD ["python", "--version"]
