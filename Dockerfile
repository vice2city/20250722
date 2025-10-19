FROM condaforge/miniforge3:latest

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    g++ \
    openjdk-11-jre-headless \
    wget \
    bzip2 \
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/20250722
RUN mamba create -n open-mmlab python=3.8 -y
SHELL ["/bin/bash", "--login", "-c"]
RUN echo "mamba activate open-mmlab" >> ~/.bashrc
ENV PATH=/opt/conda/envs/open-mmlab/bin:$PATH
ENV MAMBA_DEFAULT_ENV=open-mmlab

RUN pip install torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install torchserve==0.2.0 torch-model-archiver timm==1.0.17 transformers==4.46.3 && \
    pip install -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.12/index.html mmcv-full==1.6.2 mmdet==2.28.2
RUN mamba install conda-forge::transformers
COPY . .
RUN pip install --no-cache-dir -e . && \
    mamba clean --all -f -y

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp \
    && cp docker/serve/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R model-server /home/model-server

RUN cp docker/serve/config.properties /home/model-server/config.properties && \
    mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
