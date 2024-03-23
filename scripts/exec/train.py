import argparse
import sys
from scripts.utils.configuration import Configuration
from scripts import training
from data.datasets.CLEVRER.dataset import ClevrerDataset, ClevrerSample, RamImage
from data.datasets.ADEPT.dataset import AdeptDataset
from data.datasets.BOUNCINGBALLS.dataset import BouncingBallDataset
from scripts import evaluation_adept, evaluation_clevrer, evaluation_bb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", default="", help='path to the configuration file')
    parser.add_argument("-n", default=-1, type=int, help='optional run number')
    parser.add_argument("-load", default="", type=str, help='path to pretrained model or checkpoint')

    # Load configuration
    args = parser.parse_args(sys.argv[1:])
    cfg = Configuration(args.cfg)
    cfg.model_path = f"{cfg.model_path}"
    if args.n >= 0:
        cfg.model_path = f"{cfg.model_path}.run{args.n}"
    print(f'Training model {cfg.model_path}')

    # Load dataset
    size = (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2))
    if cfg.datatype == "clevrer":
        trainset = ClevrerDataset("./", cfg.dataset, "train", size, use_slotformer=False)
        valset   = ClevrerDataset("./", cfg.dataset, "val", size, use_slotformer=True)
    elif cfg.datatype == "adept":
        trainset = AdeptDataset("./", cfg.dataset, "train", size)
        valset   = AdeptDataset("./", cfg.dataset, "test",  size)
        valset.train = True
    elif cfg.datatype == "bouncingballs":
        trainset = BouncingBallDataset("./", cfg.dataset, "train", size,  type_name = cfg.scenario)
        valset   = BouncingBallDataset("./", cfg.dataset, "val",   size,  type_name = cfg.scenario)
    else:
        raise Exception("Dataset not supported")
    
    # Final call
    training.train_loci(cfg, trainset, valset, args.load)