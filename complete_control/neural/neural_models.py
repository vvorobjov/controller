from pydantic import BaseModel

from complete_control.utils_common.custom_types import NdArray


class PopulationSpikes(BaseModel):
    """
    Represents the spiking data and metadata for a single neuron population.
    """

    label: str
    gids: NdArray
    senders: NdArray
    times: NdArray
    population_size: int
    neuron_model: str

    class Config:
        arbitrary_types_allowed = True


class SynapseWeightRecord(BaseModel):
    source: int
    target: int
    trial: int
    weight: float
