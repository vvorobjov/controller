from nrp_core import *
from nrp_core.data.nrp_json import *

car_speed = 5


@EngineDataPack(keyword='positions', id=DataPackIdentifier('positions', 'bullet_simulator'))
@PreprocessingFunction("bullet_simulator")
def from_bullet(positions):
    
    print(positions)

    return [positions]