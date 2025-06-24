FROM docker-registry.ebrains.eu/neurorobotics/nrp/nrp-core/nrp-pybullet-ubuntu20:latest

ENV DEPS_DIR=$HOME/sim/dependencies
ENV BULLET_MUSCLE_DIR=$DEPS_DIR/bullet_muscle_sim
ENV SDF_MODELS_DIR=$HOME/sim/embodiment_sdf_models
RUN mkdir -p $BULLET_MUSCLE_DIR $SDF_MODELS_DIR

# Install bullet muscle simulation
# RUN git clone https://github.com/near-nes/bullet_muscle_sim.git $BULLET_MUSCLE_DIR
RUN git clone https://github.com/near-nes/embodiment_sdf_models.git $SDF_MODELS_DIR

# ENV PYTHONPATH=${PYTHONPATH}:${BULLET_MUSCLE_DIR}
