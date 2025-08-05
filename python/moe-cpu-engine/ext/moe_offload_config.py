class MOEOffloadConfig:
    """
    A class to manage the mapping of layer IDs to expert IDs.
    """
    def __init__(self):
        # TODO: read from config file
        self._layer_to_experts_map = {
            0: [],
            1: [],
            2: [],
            **{i: list(range(256)) for i in range(3, 61)}
        }

    def get_offload_experts(self, layer_id):
        """
        Retrieve offloaded expert IDs for a given layer ID.

        Args:
            layer_id (int): The ID of the layer.

        Returns:
            list[int]: A list of expert IDs associated with the layer.
        """
        return self._layer_to_experts_map.get(layer_id, [])


def create_offload_config():
    """
    Factory function to create a OffloadConfig instance.
    """
    return MOEOffloadConfig()