def get_plr_frequencies(latent_replay_strategy, weight_counts, layer_sizes):
    if latent_replay_strategy == "basic":
        return [1 / len(weight_counts) for _ in range(len(weight_counts))]
    elif latent_replay_strategy == "cumulative_weights":
        cumulative_updates_per_layer = [sum(weight_counts[i:]) for i in
                                        range(len(weight_counts))]
        raw_frequencies = [cumulative_updates_per_layer[0] / c for c in
                           cumulative_updates_per_layer]
        return [r / sum(raw_frequencies) for r in raw_frequencies]
    elif latent_replay_strategy == "total_weights":
        raw_frequencies = [sum(weight_counts) / c for c in
                           weight_counts]
        return [r / sum(raw_frequencies) for r in raw_frequencies]
    elif latent_replay_strategy == "input_size":
        raw_frequencies = [c / sum(layer_sizes) for c in
                           layer_sizes]
        return [r / sum(raw_frequencies) for r in raw_frequencies]
    elif latent_replay_strategy == "layer_idx":
        layer_indices = list(range(1, len(weight_counts) + 1))
        return [c / sum(layer_indices) for c in
                layer_indices]
    else:
        raise NotImplementedError()
