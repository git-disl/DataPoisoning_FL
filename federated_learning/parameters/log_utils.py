def log_model_parameter_names(logger, parameters):
    """
    :param logger: loguru.logger
    :param parameters: dict(tensor)
    """
    logger.info("Model Parameter Names: {}".format(parameters.keys()))
