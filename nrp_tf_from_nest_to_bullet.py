# This is calclated on NRP-Core side

from nrp_core import *
from nrp_core.data.nrp_json import *
import numpy as np

scale   = 350.0

@EngineDataPack(keyword='positions', id=DataPackIdentifier('positions', "bullet_simulator"))
# @EngineDataPack(keyword='brainstem_n', id=DataPackIdentifier('spikedetector_brain_stem_neg', 'nest'))
# @EngineDataPack(keyword='brainstem_p', id=DataPackIdentifier('spikedetector_brain_stem_pos', 'nest'))
@EngineDataPack(keyword='spkRate', id=DataPackIdentifier('spkRate', 'python_nest_client'))
@TransceiverFunction("bullet_simulator")
def to_bullet(positions, spkRate):
    rec_cmd = JsonDataPack("control_cmd", "bullet_simulator")


    global scale

    spkRate_net = spkRate.data["rate"]
    inputCmd = spkRate_net/ scale
    print("nrp_tf_from_nest_to_bullet: calculated inputCmd", inputCmd)
    
    # Do some stuff with pybullet action and nest data
    rec_cmd.data["act_list"] = [inputCmd * 1000]

    return [rec_cmd]