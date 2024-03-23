import argparse
import sys
from data.datasets.ADEPT.dataset import AdeptDataset
from data.datasets.BOUNCINGBALLS.dataset import BouncingBallDataset
from data.datasets.CLEVRER.dataset import ClevrerDataset, ClevrerSample, RamImage
from scripts.utils.configuration import Configuration
from scripts import evaluation_adept, evaluation_clevrer, evaluation_bb

def main(load, n, cfg):
    
    size = (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2))

     # Load dataset
    if cfg.datatype == "clevrer":
        testset  = ClevrerDataset("./", cfg.dataset, 'val', size, use_slotformer=True, evaluation=True)
        evaluation_clevrer.evaluate(cfg, testset, load, n, plot_frequency= 2, plot_first_samples = 3) # only plotting
        evaluation_clevrer.evaluate(cfg, testset, load, n, plot_first_samples = 0) # evaluation
    elif cfg.datatype == "adept":
        testset = AdeptDataset("./", cfg.dataset, 'createdown',  size)
        evaluation_adept.evaluate(cfg, testset, load, n, plot_frequency= 1, plot_first_samples = 2)
    elif cfg.datatype == "bouncingballs":
        testset   = BouncingBallDataset("./", cfg.dataset, "test", size, type_name = cfg.scenario)
        evaluation_bb.evaluate(cfg, testset, load, n, plot_first_samples = 4)
    else:
        raise Exception("Dataset not supported")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", default="", help='path to the configuration file')
    parser.add_argument("-load", default="", type=str, help='path to model')
    parser.add_argument("-n", default="", type=str, help='results name')

    # Load configuration
    args = parser.parse_args(sys.argv[1:])
    cfg = Configuration(args.cfg)
    print(f'Evaluating model {args.load}')

    main(args.load, args.n, cfg)