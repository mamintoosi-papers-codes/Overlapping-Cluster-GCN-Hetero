import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the PubMed dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description = "Run .")

    parser.add_argument("--dataset-name",
                        type = str,
                        default = "Cora",
	                help = "Dataset name (Cora, ...)")

    parser.add_argument("--ds-root",
                        type = str,
                        default = "../tmp",
	                help = "Root folder for saving datasets")

    parser.add_argument("--clustering-overlap",
                        type = bool,
                        default = False,
	                help = "Cluster overlapping")

    parser.add_argument("--membership-closeness",
                        type = float,
                        default = 0.1,
	                help = "Percentage of other nodes' memberships relative to the membership of the closest node. Default is 0.5.")

    parser.add_argument("--clustering-method",
                        nargs = "?",
                        default = "danmf",
	                help = "Clustering method for graph decomposition. Default is the DANMF procedure.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 10,
	                help = "Number of training epochs. Default is 200.")

    parser.add_argument("--num-trial",
                        type = int,
                        default = 1,
	                help = "Number of training epochs. Default is 200.")

    parser.add_argument("--seed",
                        type = int,
                        default = 42,
	                help = "Random seed for train-test split. Default is 42.")

    parser.add_argument("--dropout",
                        type = float,
                        default = 0.5,
	                help = "Dropout parameter. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.01,
	                help = "Learning rate. Default is 0.01.")

    parser.add_argument("--test-ratio",
                        type = float,
                        default = 0.1,
	                help = "Test data ratio. Default is 0.1.")

    parser.add_argument("--cluster-number",
                        type = int,
                        default = 10,
                        help = "Number of clusters extracted. Default is 10.")

    parser.add_argument("--target-node-type",
                        type = str,
                        default = None,
                        help = "Target node type used for heterogeneous node classification.")

    parser.add_argument("--hetero-projector-dim",
                        type = int,
                        default = 64,
                        help = "Unified latent dimension used to project heterogeneous node features.")

    parser.add_argument("--hetero-knn-k",
                        type = int,
                        default = 10,
                        help = "Number of nearest neighbours used to build the clustering graph for heterogeneous datasets.")

    parser.add_argument("--hetero-training-mode",
                        type = str,
                        default = "cluster",
                        choices = ["cluster", "full_batch"],
                        help = "Use clustered hetero subgraphs or a single full-batch hetero baseline run.")

    parser.set_defaults(layers = [16, 16, 16])
    
    return parser.parse_args()
