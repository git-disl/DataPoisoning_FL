from sklearn.preprocessing import StandardScaler

def apply_standard_scaler(gradients):
    scaler = StandardScaler()

    return scaler.fit_transform(gradients)
