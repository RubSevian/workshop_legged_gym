"""Checkpoint compatibility helpers for the Go2 Tumbler policy."""


def infer_history_length(checkpoint, num_observations):
    """Return the encoder history length stored in a Tumbler checkpoint.

    The first encoder layer consumes ``num_observations * history_length``.
    Reading that shape makes playback independent from the current config and
    avoids the opaque matrix-multiplication error caused by a mismatch.
    """
    candidates = (
        ("actor_state_dict", "dm_encoder.encoder.0.weight"),
        ("dm_encoder_state_dict", "encoder.0.weight"),
    )
    for state_name, weight_name in candidates:
        state_dict = checkpoint.get(state_name, {})
        weight = state_dict.get(weight_name)
        if weight is not None:
            history_width = weight.shape[1]
            if history_width % num_observations:
                raise ValueError(
                    f"Checkpoint encoder input width {history_width} is not "
                    f"divisible by num_observations={num_observations}."
                )
            return history_width // num_observations
    raise ValueError(
        "Not a supported Tumbler checkpoint: encoder first-layer weights are missing."
    )
