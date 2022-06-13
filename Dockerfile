FROM mambaorg/micromamba:0.23.0
COPY --chown=$MAMBA_USER:$MAMBA_USER conda-environments/activitysim-total.yml /tmp/env.yaml
RUN micromamba install -y -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /home/mambauser
RUN git clone https://github.com/camsys/activitysim/
RUN git -C activitysim switch with-sharrow
RUN python -m pip install -e ./activitysim
