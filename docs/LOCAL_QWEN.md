# Local Qwen reasoning in reconstruction

Set `SIMFOUNDRY_TEXT_VLM=qwen` to use a local Qwen3.5 vision-language model for
A's support-surface fallback, canonical-frame selection, object category detection,
optional crop-validity checks, front orientation, and mass/friction estimation.
The default remains Gemini. Estimated physical properties still require validation.

```bash
export SIMFOUNDRY_TEXT_VLM=qwen
export SIMFOUNDRY_QWEN_MODEL_PATH=/absolute/path/to/Qwen3.5-4B
export SIMFOUNDRY_QWEN_PYTHON=/absolute/path/to/qwen-env/bin/python
# Optional; requires bitsandbytes. Defaults to unquantized bfloat16.
export SIMFOUNDRY_QWEN_QUANTIZATION=nf4
```

The worker needs a compatible recent Transformers with Qwen3.5 support, PyTorch,
torchvision, Pillow, and accelerate. Keep these dependencies in a separate environment
from the reconstruction environments. Only local model files are loaded; the worker
sets Hugging Face offline mode. `SIMFOUNDRY_QWEN_MAX_TOKENS` (2048),
`SIMFOUNDRY_QWEN_MAX_SIDE` (1024 pixels) and `SIMFOUNDRY_QWEN_TIMEOUT_S` (600 seconds)
bound each call. Thinking is disabled for structured answers.

Optional `SIMFOUNDRY_QWEN_TEMPERATURE`, `SIMFOUNDRY_QWEN_TOP_P` and
`SIMFOUNDRY_QWEN_REPETITION_PENALTY` explicitly override caller sampling settings.
The radio capture uses 0.7, 0.8 and 1.05: greedy decoding repeated fixture names until
truncation on its stage-5 prompt. These settings are included in the cache key.
Stage 5 requests `bbox_xyxy` for Qwen, whose localization uses x/y order, while the
default Gemini prompt retains `box_2d` in y/x order. Both are existing reader formats.

Each call runs in a subprocess, releasing its model memory when it exits. This is slower
than a persistent server but avoids keeping the VLM resident between reconstruction
stages. Enough free memory is still needed while SAM3 or another stage model is resident;
the streaming scheduler does not separately budget this subprocess.
Existing CACHE_MODE / TEST_MODE use a separate `local_qwen` cache, with input image hashes,
local model metadata and generation settings in its key. Empty or truncated answers fail
and are not cached. Callers still apply their existing output parsing and validation.

Qwen3.5 emits text, not edited images. Stage 5 removal and Stage 6 image generation
continue to use their configured image backend. Do not select Qwen as a removal/upsampling
model or use its text in place of image files. FLUX Kontext is already supported for both
image stages but requires its own weights and access permission. This switch does not
replace external models inside the optional articulate-anything pipeline or background
generation. Stage 9 is not enabled by the radio reconstruction configuration.

For the local FLUX backend, `SIMFOUNDRY_FLUX_QUANTIZATION=nf4` optionally loads both
the transformer and T5 encoder in bitsandbytes NF4 with bfloat16 computation and
double quantization. The default remains unquantized. Existing CPU offloading is
retained. This trades some numerical precision for room alongside stage 5's SAM3
and depth models; validate edited images before mesh generation.
