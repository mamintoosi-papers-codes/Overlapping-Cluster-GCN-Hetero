import torch
import random
import numpy as np
from collections import defaultdict
from tqdm import trange, tqdm
from layers import StackedGCN
from layers import HeteroStackedGNN
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class ClusterGCNTrainer(object):
    """
    Training a ClusterGCN.
    """
    def __init__(self, args, clustering_machine):
        """
        :param ags: Arguments object.
        :param clustering_machine:
        """  
        self.args = args
        self.clustering_machine = clustering_machine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_model()

    def create_model(self):
        """
        Creating a StackedGCN and transferring to CPU/GPU.
        """
        if self.clustering_machine.is_hetero:
            self.model = HeteroStackedGNN(
                self.args,
                self.clustering_machine.metadata,
                self.clustering_machine.feature_count,
                self.clustering_machine.class_count,
                self.clustering_machine.target_node_type
            )
        else:
            self.model = StackedGCN(self.args, self.clustering_machine.feature_count, self.clustering_machine.class_count)
        self.model = self.model.to(self.device)

    def do_forward_pass(self, cluster):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        if self.clustering_machine.is_hetero:
            batch = self.clustering_machine.cluster_batches[cluster]
            edge_index_dict = {
                edge_type: edge_index.to(self.device)
                for edge_type, edge_index in batch["edge_index_dict"].items()
            }
            x_dict = {
                node_type: features.to(self.device)
                for node_type, features in batch["x_dict"].items()
            }
            train_nodes = batch["train_nodes"].to(self.device)
            target = batch["targets"].to(self.device).squeeze()
            predictions = self.model(x_dict, edge_index_dict)
            average_loss = torch.nn.functional.nll_loss(predictions[train_nodes], target[train_nodes])
            node_count = train_nodes.shape[0]
        else:
            edges = self.clustering_machine.sg_edges[cluster].to(self.device)
            macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
            train_nodes = self.clustering_machine.sg_train_nodes[cluster].to(self.device)
            features = self.clustering_machine.sg_features[cluster].to(self.device)
            target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
            predictions = self.model(edges, features)
            average_loss = torch.nn.functional.nll_loss(predictions[train_nodes], target[train_nodes])
            node_count = train_nodes.shape[0]
        return average_loss, node_count

    def update_average_loss(self, batch_average_loss, node_count):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster. 
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item()*node_count
        self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss/self.node_count_seen
        return average_loss

    def do_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        if self.clustering_machine.is_hetero:
            batch = self.clustering_machine.cluster_batches[cluster]
            edge_index_dict = {
                edge_type: edge_index.to(self.device)
                for edge_type, edge_index in batch["edge_index_dict"].items()
            }
            x_dict = {
                node_type: features.to(self.device)
                for node_type, features in batch["x_dict"].items()
            }
            test_nodes = batch["test_nodes"].to(self.device)
            target = batch["targets"].to(self.device).squeeze()[test_nodes]
            prediction = self.model(x_dict, edge_index_dict)[test_nodes, :]
            global_nodes = batch["global_target_nodes"].to(self.device)[test_nodes]
            return {
                "prediction": prediction,
                "target": target,
                "global_nodes": global_nodes
            }
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        test_nodes = self.clustering_machine.sg_test_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        target = target[test_nodes]
        prediction = self.model(edges, features)
        prediction = prediction[test_nodes,:]
        return {
            "prediction": prediction,
            "target": target,
            "global_nodes": None
        }

    def train(self):
        """
        Training a model.
        """
        # print("\nTraining started.")
        epochs = trange(self.args.epochs, desc = "Train Loss")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        for epoch in epochs:
            random.shuffle(self.clustering_machine.clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for cluster in self.clustering_machine.clusters:
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_forward_pass(cluster)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)
            epochs.set_description("Train Loss: %g" % round(average_loss,4))

    def test(self):
        """
        Scoring the test and printing the F-1 score.
        """
        self.model.eval()
        self.predictions = []
        self.targets = []
        if self.clustering_machine.is_hetero:
            aggregated_predictions = defaultdict(list)
            aggregated_targets = {}
            for cluster in self.clustering_machine.clusters:
                prediction_batch = self.do_prediction(cluster)
                for node_id, node_target, node_prediction in zip(
                    prediction_batch["global_nodes"].cpu().detach().numpy().tolist(),
                    prediction_batch["target"].cpu().detach().numpy().tolist(),
                    prediction_batch["prediction"].cpu().detach().numpy()
                ):
                    aggregated_targets[node_id] = node_target
                    aggregated_predictions[node_id].append(node_prediction)
            ordered_nodes = sorted(aggregated_predictions.keys())
            self.targets = np.array([aggregated_targets[node_id] for node_id in ordered_nodes])
            self.predictions = np.vstack([
                np.asarray(aggregated_predictions[node_id]).mean(axis=0)
                for node_id in ordered_nodes
            ]).argmax(1)
        else:
            for cluster in self.clustering_machine.clusters:
                prediction_batch = self.do_prediction(cluster)
                self.predictions.append(prediction_batch["prediction"].cpu().detach().numpy())
                self.targets.append(prediction_batch["target"].cpu().detach().numpy())
            self.targets = np.concatenate(self.targets)
            self.predictions = np.concatenate(self.predictions).argmax(1)
        score = f1_score(self.targets, self.predictions, average="micro")
        return score
        # print("\nF-1 score: {:.4f}".format(score))
