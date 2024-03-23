import argparse
import sys
from data.datasets.ADEPT.dataset import AdeptDataset
from scripts.evaluation_adept_baselines import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-load", default="", type=str, help='path to savi slots')
    parser.add_argument("-n", default="", type=str, help='results name')
    parser.add_argument("-model", default="", type=str, help='model, either savi or gswm')

    # Load configuration
    args = parser.parse_args(sys.argv[1:])
    print(f'Evaluating savi slots {args.load}')

    # Load dataset
    testset = AdeptDataset("./", 'adept', 'createdown',  (30 * 2**(2*2), 20 * 2**(2*2)))
    evaluate(testset, args.load, args.n, args.model, plot_frequency= 1, plot_first_samples = 2)