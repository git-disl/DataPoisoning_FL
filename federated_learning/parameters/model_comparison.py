import torch

def compare_models(logger, model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if not torch.equal(key_item_1[1], key_item_2[1]):
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                logger.error('Mismtach found at {}', key_item_1[0])
                logger.debug("Model 1 value: {}", str(key_item_1[1]))
                logger.debug("Model 2 value: {}", str(key_item_2[1]))
            else:
                raise Exception
    if models_differ == 0:
        logger.info('Models match perfectly!')
