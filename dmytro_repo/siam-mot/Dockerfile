FROM aicrowd/base-images:py37-cuda11-torch171-maskrcnn

# Set up AIcrowd user for evaluation
ARG NB_USER=aicrowd
ARG NB_UID=1001
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

ARG REPO_DIR=${HOME}
ENV REPO_DIR ${REPO_DIR}
WORKDIR ${REPO_DIR}

# Install additional dependencies
COPY siam-mot/requirements_exact.txt requirements_exact.txt
RUN pip install -r requirements_exact.txt --use-deprecated=legacy-resolver

COPY siam-mot/requirements_additional.txt requirements_additional.txt
RUN pip install -r requirements_additional.txt

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Add siam-mot related PATH
ENV PATH ${HOME}/.local/bin:${REPO_DIR}/.local/bin:${PATH}
ENV PATH="${PATH}:/home/aicrowd/siam-mot"
ENV PYTHONPATH="${PYTHONPATH}:/home/aicrowd/siam-mot"

# Copy current directory as submission
COPY . ${REPO_DIR}

# Entry point for evaluation
# You can skip in case you are using this Dockerfile for another purpose
RUN chown -R ${NB_USER}:${NB_USER} ${REPO_DIR}
USER ${NB_USER}
ENTRYPOINT ["/home/aicrowd/run.sh"]
