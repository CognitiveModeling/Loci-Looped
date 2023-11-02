import argparse
import sys
from data.datasets.ADEPT.dataset import AdeptDataset
from data.datasets.CLEVRER.dataset import ClevrerDataset, ClevrerSample, RamImage
from scripts.utils.configuration import Configuration
from scripts import evaluation_adept, evaluation_clevrer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", default="", help='path to the configuration file')
    parser.add_argument("-load", default="", type=str, help='path to model')
    parser.add_argument("-n", default="", type=str, help='results name')

    # Load configuration
    args = parser.parse_args(sys.argv[1:])
    cfg = Configuration(args.cfg)
    print(f'Evaluating model {args.load}')

    # Load dataset
    if cfg.datatype == "clevrer":
        testset  = ClevrerDataset("./", cfg.dataset, 'val', (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)), use_slotformer=True, evaluation=True)
        evaluation_clevrer.evaluate(cfg, testset, args.load, args.n, plot_frequency= 2, plot_first_samples = 3) # only plotting
        evaluation_clevrer.evaluate(cfg, testset, args.load, args.n, plot_first_samples = 0) # evaluation
    elif cfg.datatype == "adept":
        testset = AdeptDataset("./", cfg.dataset, 'createdown',  (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)))
        evaluation_adept.evaluate(cfg, testset, args.load, args.n, plot_frequency= 1, plot_first_samples = 2)
    else:
        raise Exception("Dataset not supported")