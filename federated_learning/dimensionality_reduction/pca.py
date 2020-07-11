from sklearn.decomposition import PCA

def calculate_pca_of_gradients(logger, gradients, num_components):
    pca = PCA(n_components=num_components)

    logger.info("Computing {}-component PCA of gradients".format(num_components))

    return pca.fit_transform(gradients)
