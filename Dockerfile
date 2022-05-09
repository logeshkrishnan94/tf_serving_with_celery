ARG MODEL_NAME=latest

FROM tensorflow/serving:latest-devel as build_image

RUN echo ${MODEL_NAME}

RUN apt-get update && apt-get install wget -y

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Expose ports
# gRPC
EXPOSE 8500

# REST
EXPOSE 8501

RUN tensorflow_model_server --version

RUN mkdir /usr/src/models

COPY ./${MODEL_NAME} /usr/src/models/${MODEL_NAME}

RUN ls /usr/src/models

RUN tensorflow_model_server --version

CMD ["tensorflow_model_server", "--rest_api_port=8501", "--model_name=${MODEL_NAME}", "--model_base_path=/usr/src/models/${MODEL_NAME}"]
