FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    MODEL_DEVICE=cpu \
    MODEL_CKPT=model.pt

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 \
    fastapi==0.115.5 \
    uvicorn==0.32.1 \
    numpy==2.1.3 \
    pydantic==2.10.2

COPY . /app

EXPOSE 7860

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]

