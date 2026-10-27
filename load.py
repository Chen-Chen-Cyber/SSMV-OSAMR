from project_paths import configured_path
import pickle
import torch
import numpy as np
import torch.nn.functional as F


def load_data(filename=configured_path("RADIO_SSL_RML2016A", "data/raw/RML2016.10a_dict.pkl")):

    Xd = pickle.load(open(filename, 'rb'), encoding='iso-8859-1')
    mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1]]

    X = []

    lbl = []

    train_idx = []
    val_idx = []
    Y_train_str = []
    np.random.seed(2016)
    a = 0

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])  # ndarray(1000,2,128)
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
            train_idx += list(np.random.choice(range(a * 1000, (a + 1) * 1000), size=600, replace=False))
            val_idx += list(np.random.choice(list(set(range(a * 1000, (a + 1) * 1000)) - set(train_idx)), size=200,
                                             replace=False))
            a += 1
    X = np.vstack(
        X)
    n_examples = X.shape[0]

    test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    def to_onehot(yy):

        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1
    Y_train_mode = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
    for number in Y_train_mode:
        Y_train_str.append(str(mods[number]))
    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_val = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
    return (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx),Y_train_str


def load_data_2016b(
    filename=configured_path("RADIO_SSL_RML2016B", "data/raw/RML2016.10b.dat"),
    seed=2016,
):

    with open(filename, "rb") as file:
        Xd = pickle.load(file, encoding="iso-8859-1")


    mods = sorted({key[0] for key in Xd.keys()})
    snrs = sorted({key[1] for key in Xd.keys()})

    X_blocks = []
    lbl = []

    train_idx = []
    val_idx = []
    test_idx = []

    rng = np.random.RandomState(seed)
    offset = 0

    for mod in mods:
        for snr in snrs:
            block = np.asarray(Xd[(mod, snr)])
            n_samples = block.shape[0]

            X_blocks.append(block)
            lbl.extend([(mod, snr)] * n_samples)


            local_indices = rng.permutation(n_samples)

            n_train = int(n_samples * 0.60)
            n_val = int(n_samples * 0.20)

            local_train = local_indices[:n_train]
            local_val = local_indices[n_train:n_train + n_val]
            local_test = local_indices[n_train + n_val:]


            train_idx.extend((local_train + offset).tolist())
            val_idx.extend((local_val + offset).tolist())
            test_idx.extend((local_test + offset).tolist())

            offset += n_samples

    X = np.vstack(X_blocks)


    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    def to_onehot(class_indices):
        class_indices = np.asarray(class_indices, dtype=np.int64)
        onehot = np.zeros(
            (len(class_indices), len(mods)),
            dtype=np.float32,
        )
        onehot[np.arange(len(class_indices)), class_indices] = 1.0
        return onehot

    train_labels = np.asarray(
        [mods.index(lbl[index][0]) for index in train_idx],
        dtype=np.int64,
    )
    val_labels = np.asarray(
        [mods.index(lbl[index][0]) for index in val_idx],
        dtype=np.int64,
    )
    test_labels = np.asarray(
        [mods.index(lbl[index][0]) for index in test_idx],
        dtype=np.int64,
    )

    Y_train = to_onehot(train_labels)
    Y_val = to_onehot(val_labels)
    Y_test = to_onehot(test_labels)

    Y_train_str = [str(mods[index]) for index in train_labels]

    print("Modulations:", mods)
    print("SNRs:", snrs)
    print("Full data shape:", X.shape)
    print("Training data shape:", X_train.shape)
    print("Validation data shape:", X_val.shape)
    print("Test data shape:", X_test.shape)

    return (
        (mods, snrs, lbl),
        (X_train, Y_train),
        (X_val, Y_val),
        (X_test, Y_test),
        (train_idx, val_idx, test_idx),
        Y_train_str,
    )


if __name__ == "__main__":
    (
        (mods, snrs, lbl),
        (X_train, Y_train),
        (X_val, Y_val),
        (X_test, Y_test),
        (train_idx, val_idx, test_idx),
        Y_train_str,
    ) = load_data_2016b()
