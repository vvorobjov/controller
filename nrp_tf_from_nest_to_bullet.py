from nrp_core import *
from nrp_core.data.nrp_json import *
import numpy as np

# Simulation parameters
time_buffer = 10 # [ms]
w = 1.0625540740843757
scale   = 350.0
nNeurons = 50
# This should be set to the minimal engine time step
comm_time_step = 20 # [ms]
current_time = 0

def computeRate(spikes, w, nNeurons, current_time, buffer_size):
    rate = 0
    if len(spikes) > 0 and current_time > buffer_size:
        np_spikes = np.array(spikes)
        count = w * np.sum((np_spikes >= (current_time-buffer_size)) & (np_spikes < current_time))
        rate = (count/((buffer_size)*nNeurons))
    return rate

@EngineDataPack(keyword='positions', id=DataPackIdentifier('positions', "bullet_simulator"))
@EngineDataPack(keyword='brainstem_n', id=DataPackIdentifier('spikedetector_brain_stem_neg', 'nest'))
@EngineDataPack(keyword='brainstem_p', id=DataPackIdentifier('spikedetector_brain_stem_pos', 'nest'))
@TransceiverFunction("bullet_simulator")
def to_bullet(positions, brainstem_n, brainstem_p):
    rec_cmd = JsonDataPack("control_cmd", "bullet_simulator")

    global comm_time_step
    global current_time
    global time_buffer
    global w
    global scale
    global nNeurons

    # Get data from NEST
    spikes_p = brainstem_p.data[0]['events']['times']
    spikes_n = brainstem_n.data[0]['events']['times']
    # print("spikes_p", spikes_p)
    # print("spikes_n", spikes_n)

    spkRate_pos = computeRate(spikes_p, w, nNeurons, current_time, time_buffer)
    spkRate_neg = computeRate(spikes_n, w, nNeurons, current_time, time_buffer)
    # print("spkRate_pos", spkRate_pos)
    # print("spkRate_neg", spkRate_neg)
    spkRate_net = spkRate_pos - spkRate_neg
    inputCmd = spkRate_net/ scale
    print("to_bullet: calculated inputCmd", inputCmd)

    current_time = current_time + comm_time_step
    print("to_bullet: calculated current_time", current_time)
    
    # Do some stuff with pybullet action and nest data
    rec_cmd.data["act_list"] = [inputCmd * 1000]

    return [rec_cmd]