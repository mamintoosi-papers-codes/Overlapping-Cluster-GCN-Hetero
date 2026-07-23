import torch


def _node_in_cluster(cluster_membership, node_id, cluster, overlapping):
    membership = cluster_membership[node_id]
    if overlapping:
        return cluster in membership
    return membership == cluster


def _build_target_masks(selected_target_nodes, train_mask, test_mask):
    train_nodes = [index for index, node_id in enumerate(selected_target_nodes) if bool(train_mask[node_id])]
    test_nodes = [index for index, node_id in enumerate(selected_target_nodes) if bool(test_mask[node_id])]
    return torch.LongTensor(train_nodes), torch.LongTensor(test_nodes)


def build_hetero_cluster_partitions(args,
                                    clusters,
                                    cluster_membership,
                                    projected_x_dict,
                                    edge_index_dict,
                                    target,
                                    train_mask,
                                    test_mask,
                                    target_node_type,
                                    local_to_global):
    """
    Build per-cluster heterogeneous sub-graphs.
    """
    cluster_batches = {}
    active_clusters = []
    all_global_nodes = set(torch.cat(list(local_to_global.values())).tolist())
    target_global_ids = set(local_to_global[target_node_type].tolist())

    if args.hetero_training_mode == "full_batch":
        clusters = [0]
        cluster_membership = {node_id: 0 for node_id in sorted(all_global_nodes)}

    for cluster in clusters:
        if args.hetero_training_mode == "full_batch":
            seed_nodes = set(all_global_nodes)
        else:
            seed_nodes = {
                node_id for node_ids in local_to_global.values() for node_id in node_ids.tolist()
                if _node_in_cluster(cluster_membership, node_id, cluster, args.clustering_overlap)
            }

        target_seed_nodes = sorted(seed_nodes.intersection(target_global_ids))
        if not target_seed_nodes:
            continue

        included_nodes = set(seed_nodes)
        for (source_type, _, target_type), edge_index in edge_index_dict.items():
            source_global = local_to_global[source_type][edge_index[0]].tolist()
            target_global = local_to_global[target_type][edge_index[1]].tolist()
            for src_global, dst_global in zip(source_global, target_global):
                if src_global in seed_nodes or dst_global in seed_nodes:
                    included_nodes.add(src_global)
                    included_nodes.add(dst_global)

        nodes_by_type = {}
        local_id_maps = {}
        for node_type, node_ids in local_to_global.items():
            selected_nodes = [node_id for node_id in node_ids.tolist() if node_id in included_nodes]
            if not selected_nodes:
                continue
            global_to_local_id = {global_id: local_id for local_id, global_id in enumerate(node_ids.tolist())}
            local_ids = [global_to_local_id[node_id] for node_id in selected_nodes]
            nodes_by_type[node_type] = torch.LongTensor(local_ids)
            local_id_maps[node_type] = {local_id: index for index, local_id in enumerate(local_ids)}

        if target_node_type not in nodes_by_type:
            continue

        selected_target_nodes = nodes_by_type[target_node_type].tolist()
        train_nodes, test_nodes = _build_target_masks(selected_target_nodes, train_mask, test_mask)
        if train_nodes.numel() == 0 or test_nodes.numel() == 0:
            continue

        cluster_x_dict = {
            node_type: projected_x_dict[node_type][node_indices]
            for node_type, node_indices in nodes_by_type.items()
        }

        cluster_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            source_type, _, target_type = edge_type
            if source_type not in nodes_by_type or target_type not in nodes_by_type:
                continue
            source_nodes = set(nodes_by_type[source_type].tolist())
            target_nodes = set(nodes_by_type[target_type].tolist())
            remapped_edges = []
            for src_local, dst_local in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                if src_local in source_nodes and dst_local in target_nodes:
                    remapped_edges.append([
                        local_id_maps[source_type][src_local],
                        local_id_maps[target_type][dst_local]
                    ])
            if remapped_edges:
                cluster_edge_index_dict[edge_type] = torch.LongTensor(remapped_edges).t().contiguous()

        cluster_batches[cluster] = {
            "x_dict": cluster_x_dict,
            "edge_index_dict": cluster_edge_index_dict,
            "target_nodes": torch.LongTensor(selected_target_nodes),
            "global_target_nodes": local_to_global[target_node_type][selected_target_nodes],
            "train_nodes": train_nodes,
            "test_nodes": test_nodes,
            "targets": target[selected_target_nodes].long()
        }
        active_clusters.append(cluster)

    return active_clusters, cluster_batches
