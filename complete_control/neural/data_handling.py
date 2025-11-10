from pathlib import Path

import structlog
from mpi4py.MPI import Comm
from neural.result_models import NeuralResultManifest
from neural.Controller import PopulationBlocks
from neural.nest_adapter import nest
from neural.neural_models import (
    SynapseBlock,
    SynapseRecording,
)

_log: structlog.stdlib.BoundLogger = structlog.get_logger(str(__file__))


def collapse_files(
    dir: Path,
    pop_blocks: PopulationBlocks,
    comm: Comm = None,
):
    """
    Collapses multiple ASCII recording files from different processes into single files per population.
    TODO decide how to handle non-ascii popviews: fail or ignore?
    Parameters
    ----------
    dir : str
        Directory path containing the recording files
    pop_blocks : PopulationBlocks
    comm : Comm
        Comm on which to barrier() on
    Notes
    -----
    Files are processed only by rank 0 process. For each population, files starting with
    the population name are combined, duplicates are removed, and original files are deleted.
    """

    controller_rec = cerebhandler_rec = cereb_rec = None
    use_cerebellum = False

    controller_rec = pop_blocks.controller.to_recording(dir, comm)

    if pop_blocks.cerebellum_handler:
        use_cerebellum = True
        cerebhandler_rec = pop_blocks.cerebellum_handler.to_recording(dir, comm)
        cereb_rec = pop_blocks.cerebellum.to_recording(dir, comm)

    if comm is not None:
        nest.SyncProcesses()

    return NeuralResultManifest(
        controller=controller_rec,
        cerebellum_handler=cerebhandler_rec,
        cerebellum=cereb_rec,
        use_cerebellum=use_cerebellum,
    )


def save_conn_weights(weights_history: dict, dir: Path, filename_prefix: str):
    """
    merge SynapseRecording objects with the same (source, target, type, trials_recorded),
    concatenate their weight_history, and save as a JSON array.
    """
    for (source_pop, target_pop), inner in weights_history.items():
        label = f"{source_pop.label}>{target_pop.label}"
        recs = []
        for (
            (source_neur, target_neur, synapse_id, synapse_model),
            weights,
        ) in inner.items():
            recs.append(
                SynapseRecording(
                    source=source_neur,
                    target=target_neur,
                    syn_id=synapse_id,
                    syn_type=synapse_model,
                    weight_history=weights,
                )
            )
        s = SynapseBlock(
            source_pop_label=source_pop.label,
            target_pop_label=target_pop.label,
            synapse_recordings=recs,
        )
        rec_path = dir / f"{filename_prefix}-{label}.json"
        with open(rec_path, "w") as f:
            json_array = s.model_dump_json(indent=2)
            f.write(json_array)
