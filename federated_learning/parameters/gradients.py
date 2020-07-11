import numpy

def calculate_model_gradient(logger, model_1, model_2):
    """
    Calculates the gradient (parameter difference) between two Torch models.

    :param logger: loguru.logger
    :param model_1: torch.nn
    :param model_2: torch.nn
    """
    model_1_parameters = list(dict(model_1.state_dict()))
    model_2_parameters = list(dict(model_2.state_dict()))

    return calculate_parameter_gradients(logger, model_1_parameters, model_2_parameters)

def calculate_parameter_gradients(logger, params_1, params_2):
    """
    Calculates the gradient (parameter difference) between two sets of Torch parameters.

    :param logger: loguru.logger
    :param model_1: dict
    :param model_2: dict
    """
    logger.debug("Shape of model_1_parameters: {}".format(str(len(params_1))))
    logger.debug("Shape of model_2_parameters: {}".format(str(len(params_2))))

    return numpy.array([x for x in numpy.subtract(params_1, params_2)])
