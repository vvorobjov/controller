import sys

sys.path.append("/sim/controller/complete_control")


from complete_control.neural.neural_models import SynapseBlock
from complete_control.neural.result_models import NeuralResultManifest
from complete_control.config.ResultMeta import ResultMeta

final_id = "20251120_173722_w82r-singletrial"


def gather_ids(id):
    meta = ResultMeta.from_id(id)
    if meta.parent is None or len(meta.parent) == 0:
        return []
    return [id, *gather_ids(meta.parent)]


def main():
    ids = gather_ids(final_id)
    for i in range(10):
        for id in ids:
            meta = ResultMeta.from_id(id)
            neural = meta.load_neural()
            with open(neural.weights[0], "r") as f:
                block = SynapseBlock.model_validate_json(f.read())
            print(
                f"{meta.id}->{block.source_pop_label}-{block.target_pop_label}[0]:{block.synapse_recordings[i].weight}"
            )
        print("\n")


if __name__ == "__main__":
    main()
