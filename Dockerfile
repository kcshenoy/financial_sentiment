# Production Dockerfile — AWS Lambda (Python 3.11)
# Build: docker build -t financial-sentiment .
# Push:  docker tag financial-sentiment <ecr-uri>:latest && docker push <ecr-uri>:latest

FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

LABEL maintainer="randomstuff.kcs@gmail.com" \
      description="Financial Sentiment Analyzer — DistilBERT on Lambda" \
      version="1.0.0"

# Install CPU-only PyTorch wheel from the official index; avoids pulling the
# full CUDA build (~2 GB) that pip resolves by default from PyPI.
COPY requirements.txt .

RUN dnf install -y findutils && dnf clean all && \
    pip install --no-cache-dir --upgrade pip --root-user-action=ignore && \
    pip install --no-cache-dir --root-user-action=ignore \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt && \
    find /var/lang/lib/python3.12/site-packages -name "tests" -type d -prune -exec rm -rf {} + && \
    find /var/lang/lib/python3.12/site-packages -name "*.pyi" -delete

# Copy fine-tuned model artifacts into the container.
# Run convert_to_onnx.py first so the onnx/ sub-directory is present.
#
# Expected layout:
#   model/
#     config.json
#     tokenizer_config.json
#     vocab.txt
#     special_tokens_map.json
#     pytorch_model.bin  (or model.safetensors)
#     onnx/
#       model_quantized.onnx   ← preferred (INT8, ~half size)
#       model.onnx             ← fallback (FP32)
COPY model/ ${LAMBDA_TASK_ROOT}/model/

# Application code
COPY app.py ${LAMBDA_TASK_ROOT}/

# Lambda handler: <module>.<callable>
CMD ["app.handler"]
