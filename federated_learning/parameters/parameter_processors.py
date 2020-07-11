def get_layer_parameters(parameters, layer_name):
    """
    Get a specific layer of parameters from a parameters object.

    :param parameters: dict(tensor)
    :param layer_name: string
    """
    return parameters[layer_name]
