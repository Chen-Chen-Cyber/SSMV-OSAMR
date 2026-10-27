from project_paths import configured_path
import os
import ast
import h5py
import numpy as np
from tqdm import tqdm

from WVD import build_open_set_train_test,split_open_test_by_snr


def convert_radcom_hdf5_to_npy(
    filename=configured_path("RADIO_SSL_RADCOM_HDF5", "data/raw/RadComOta2.45GHz.hdf5"),
    save_dir=configured_path("RADIO_SSL_RADCOM_NPY", "data/radcom_npy"),
    radar_only=False,
):
    "Convert RadCom HDF5 samples to contiguous NPY arrays. Keys contain (modulation, signal_type, snr, sample_id), for example ('AM-DSB', 'AM radio', '0', '0') or ('PULSED', 'Ground mapping', '28', '6813'). Convert each (256,) sample to (2, 128). Save RadCom_X.npy, RadCom_modulation.npy, RadCom_signal.npy, RadCom_snr.npy and RadCom_sample_id.npy. radar_only=False retains all samples; True retains radar-related signal types only."

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    radar_keywords = [
        "radar",
        "airborne",
        "ground",
        "mti",
        "altimeter",
        "range",
        "mapping",
    ]


    with h5py.File(
        filename,
        "r"
    ) as f:

        keys = list(
            f.keys()
        )

        print(
            "Total HDF5 samples:",
            len(keys)
        )

        selected = []

        print(
            "Parsing metadata..."
        )

        for key in tqdm(
            keys,
            desc="Parse keys"
        ):

            info = ast.literal_eval(
                key
            )

            if len(info) != 4:
                raise ValueError(
                    f"Unexpected key format: {key}"
                )

            modulation = str(
                info[0]
            )

            signal_type = str(
                info[1]
            )

            snr = float(
                info[2]
            )

            sample_id = str(
                info[3]
            )


            if radar_only:

                signal_lower = (
                    signal_type.lower()
                )

                is_radar = any(
                    word.lower()
                    in signal_lower
                    for word in radar_keywords
                )

                if not is_radar:
                    continue

            selected.append(
                (
                    key,
                    modulation,
                    signal_type,
                    snr,
                    sample_id
                )
            )


    N = len(
        selected
    )

    print(
        "Selected samples:",
        N
    )

    if N == 0:
        raise ValueError(
            "No samples selected."
        )


    X = np.empty(
        (
            N,
            2,
            128
        ),
        dtype=np.float32
    )

    modulation_array = np.empty(
        N,
        dtype="<U32"
    )

    signal_array = np.empty(
        N,
        dtype="<U64"
    )

    snr_array = np.empty(
        N,
        dtype=np.float32
    )

    sample_id_array = np.empty(
        N,
        dtype="<U32"
    )


    print(
        "Loading IQ samples..."
    )

    with h5py.File(
        filename,
        "r"
    ) as f:

        for i, (
            key,
            modulation,
            signal_type,
            snr,
            sample_id
        ) in enumerate(
            tqdm(
                selected,
                desc="Convert RadCom"
            )
        ):

            sample = f[
                key
            ][()]


            if sample.shape != (
                256,
            ):
                raise ValueError(
                    f"Expected sample shape (256,), "
                    f"but got {sample.shape}, "
                    f"key={key}"
                )


            X[
                i,
                0,
                :
            ] = sample[
                :128
            ]

            X[
                i,
                1,
                :
            ] = sample[
                128:
            ]

            modulation_array[
                i
            ] = modulation

            signal_array[
                i
            ] = signal_type

            snr_array[
                i
            ] = snr

            sample_id_array[
                i
            ] = sample_id


    X_path = os.path.join(
        save_dir,
        "RadCom_X.npy"
    )

    modulation_path = os.path.join(
        save_dir,
        "RadCom_modulation.npy"
    )

    signal_path = os.path.join(
        save_dir,
        "RadCom_signal.npy"
    )

    snr_path = os.path.join(
        save_dir,
        "RadCom_snr.npy"
    )

    sample_id_path = os.path.join(
        save_dir,
        "RadCom_sample_id.npy"
    )

    print(
        "\nSaving NPY files..."
    )

    np.save(
        X_path,
        X
    )

    np.save(
        modulation_path,
        modulation_array
    )

    np.save(
        signal_path,
        signal_array
    )

    np.save(
        snr_path,
        snr_array
    )

    np.save(
        sample_id_path,
        sample_id_array
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "RadCom conversion finished"
    )

    print(
        "=" * 80
    )

    print(
        "X shape:",
        X.shape
    )

    print(
        "X dtype:",
        X.dtype
    )

    print(
        "Example sample shape:",
        X[0].shape
    )

    print(
        "\nSaved files:"
    )

    print(
        X_path
    )

    print(
        modulation_path
    )

    print(
        signal_path
    )

    print(
        snr_path
    )

    print(
        sample_id_path
    )

    print(
        "=" * 80
    )

    return (
        X_path,
        modulation_path,
        signal_path,
        snr_path,
        sample_id_path
    )


def load_data_radcom_npy(
    data_dir=configured_path("RADIO_SSL_RADCOM_NPY", "data/radcom_npy"),
    seed=2016,
    task="signal",
    train_ratio=0.60,
    val_ratio=0.20,
    mmap_mode=None,
):
    "Load converted RadCom NPY arrays. task='signal' selects signal-type labels; task='modulation' selects modulation labels. mmap_mode=None loads all arrays, whereas 'r' uses read-only memory mapping. Split each class-SNR group independently into 60% training, 20% validation and 20% testing. Return ((classes, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx), Y_train_str)."

    rng = np.random.RandomState(
        seed
    )


    X_path = os.path.join(
        data_dir,
        "RadCom_X.npy"
    )

    modulation_path = os.path.join(
        data_dir,
        "RadCom_modulation.npy"
    )

    signal_path = os.path.join(
        data_dir,
        "RadCom_signal.npy"
    )

    snr_path = os.path.join(
        data_dir,
        "RadCom_snr.npy"
    )

    sample_id_path = os.path.join(
        data_dir,
        "RadCom_sample_id.npy"
    )


    required_files = [
        X_path,
        modulation_path,
        signal_path,
        snr_path,
        sample_id_path
    ]

    for path in required_files:

        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                f"File not found: {path}"
            )


    print(
        "Loading RadCom NPY..."
    )

    X = np.load(
        X_path,
        mmap_mode=mmap_mode
    )

    modulation_raw = np.load(
        modulation_path
    )

    signal_raw = np.load(
        signal_path
    )

    snr_raw = np.load(
        snr_path
    ).astype(
        np.float32
    )

    sample_id_raw = np.load(
        sample_id_path
    )


    if (
        X.ndim != 3
        or X.shape[1] != 2
        or X.shape[2] != 128
    ):
        raise ValueError(
            f"Expected X shape [N,2,128], "
            f"but got {X.shape}"
        )


    if task == "signal":

        target_raw = (
            signal_raw
        )

    elif task == "modulation":

        target_raw = (
            modulation_raw
        )

    else:

        raise ValueError(
            'task must be "signal" '
            'or "modulation"'
        )


    classes = sorted(
        np.unique(
            target_raw
        ).tolist()
    )

    snrs = sorted(
        np.unique(
            snr_raw
        ).tolist()
    )

    class_to_index = {
        cls: idx
        for idx, cls
        in enumerate(
            classes
        )
    }


    labels_integer = np.asarray(
        [
            class_to_index[
                label
            ]
            for label in target_raw
        ],
        dtype=np.int64
    )


    lbl = [
        (
            target_raw[i],
            float(
                snr_raw[i]
            )
        )
        for i in range(
            len(target_raw)
        )
    ]


    train_idx = []
    val_idx = []
    test_idx = []

    for class_id in range(
        len(classes)
    ):

        for snr in snrs:

            indices = np.flatnonzero(
                (
                    labels_integer
                    == class_id
                )
                &
                (
                    snr_raw
                    == snr
                )
            )

            n_samples = len(
                indices
            )

            if n_samples == 0:
                continue

            indices = rng.permutation(
                indices
            )

            n_train = int(
                n_samples
                * train_ratio
            )

            n_val = int(
                n_samples
                * val_ratio
            )

            train_part = indices[
                :n_train
            ]

            val_part = indices[
                n_train:
                n_train + n_val
            ]

            test_part = indices[
                n_train + n_val:
            ]

            train_idx.extend(
                train_part.tolist()
            )

            val_idx.extend(
                val_part.tolist()
            )

            test_idx.extend(
                test_part.tolist()
            )


    train_idx = np.asarray(
        train_idx,
        dtype=np.int64
    )

    val_idx = np.asarray(
        val_idx,
        dtype=np.int64
    )

    test_idx = np.asarray(
        test_idx,
        dtype=np.int64
    )


    rng.shuffle(
        train_idx
    )

    rng.shuffle(
        val_idx
    )

    rng.shuffle(
        test_idx
    )


    X_train = np.asarray(
        X[
            train_idx
        ],
        dtype=np.float32
    )

    X_val = np.asarray(
        X[
            val_idx
        ],
        dtype=np.float32
    )

    X_test = np.asarray(
        X[
            test_idx
        ],
        dtype=np.float32
    )

    train_labels = labels_integer[
        train_idx
    ]

    val_labels = labels_integer[
        val_idx
    ]

    test_labels = labels_integer[
        test_idx
    ]


    eye = np.eye(
        len(classes),
        dtype=np.float32
    )

    Y_train = eye[
        train_labels
    ]

    Y_val = eye[
        val_labels
    ]

    Y_test = eye[
        test_labels
    ]


    Y_train_str = [
        classes[
            index
        ]
        for index
        in train_labels
    ]


    print(
        "\n"
        + "=" * 80
    )

    print(
        "RadCom NPY Dataset"
    )

    print(
        "=" * 80
    )

    print(
        "Task:",
        task
    )

    print(
        "\nClasses:"
    )

    for i, cls in enumerate(
        classes
    ):

        print(
            f"{i}: {cls}"
        )

    print(
        "\nNumber of classes:",
        len(classes)
    )

    print(
        "\nSNRs:",
        snrs
    )

    print(
        "\nNumber of SNRs:",
        len(snrs)
    )

    print(
        "\nFull data shape:",
        X.shape
    )

    print(
        "Training data shape:",
        X_train.shape
    )

    print(
        "Validation data shape:",
        X_val.shape
    )

    print(
        "Test data shape:",
        X_test.shape
    )

    print(
        "\nY_train shape:",
        Y_train.shape
    )

    print(
        "Y_val shape:",
        Y_val.shape
    )

    print(
        "Y_test shape:",
        Y_test.shape
    )

    print(
        "\nExample sample shape:",
        X_train[0].shape
    )

    print(
        "=" * 80
    )


    return (
        (
            classes,
            snrs,
            lbl
        ),
        (
            X_train,
            Y_train
        ),
        (
            X_val,
            Y_val
        ),
        (
            X_test,
            Y_test
        ),
        (
            train_idx,
            val_idx,
            test_idx
        ),
        Y_train_str
    )

def sequence_transform(data):
    'Convert raw IQ data of shape (N, 2, 128) to features of shape (N, 10, 128).'
    data = np.asarray(data, dtype=np.float32)

    i_data = data[:, 0, :]
    q_data = data[:, 1, :]

    complex_data = i_data + 1j * q_data

    amplitude = np.abs(complex_data)
    phase = np.angle(complex_data)

    sin_feature = amplitude * np.sin(phase)
    cos_feature = amplitude * np.cos(phase)

    fft_data = np.fft.fft(complex_data, axis=-1)

    fft_real = fft_data.real
    fft_imag = fft_data.imag
    fft_magnitude = np.log1p(np.abs(fft_data))
    fft_phase = np.angle(fft_data)

    features = np.stack(
        [
            i_data,
            q_data,
            amplitude,
            phase,
            sin_feature,
            cos_feature,
            fft_real,
            fft_imag,
            fft_magnitude,
            fft_phase
        ],
        axis=1
    )

    return features.astype(np.float32)


import os

import numpy as np


def load_data_radcom_npy_grouped(
    data_dir=configured_path("RADIO_SSL_RADCOM_NPY", "data/radcom_npy"),
    seed=2016,
    task="signal",
    train_ratio=0.60,
    val_ratio=0.20,
    mmap_mode=None,
    require_complete_grid=True,
):
    """Load RadCom NPY data and split it by sample_id.

    All records sharing the same sample_id are assigned to the same split.
    This prevents related records from different classes or SNR conditions
    from appearing in both training and evaluation sets.

    The return format matches the original ``load_data_radcom_npy`` function.
    """

    if task not in {"signal", "modulation"}:
        raise ValueError('task must be "signal" or "modulation"')

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    paths = {
        "X": os.path.join(data_dir, "RadCom_X.npy"),
        "modulation": os.path.join(data_dir, "RadCom_modulation.npy"),
        "signal": os.path.join(data_dir, "RadCom_signal.npy"),
        "snr": os.path.join(data_dir, "RadCom_snr.npy"),
        "sample_id": os.path.join(data_dir, "RadCom_sample_id.npy"),
    }

    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {name} file: {path}")

    X = np.load(paths["X"], mmap_mode=mmap_mode)
    modulation_raw = np.load(paths["modulation"])
    signal_raw = np.load(paths["signal"])
    snr_raw = np.load(paths["snr"]).astype(np.float32, copy=False)
    sample_id_raw = np.load(paths["sample_id"])

    arrays = {
        "modulation": modulation_raw,
        "signal": signal_raw,
        "snr": snr_raw,
        "sample_id": sample_id_raw,
    }

    if X.ndim != 3 or X.shape[1:] != (2, 128):
        raise ValueError(f"Expected X shape [N, 2, 128], got {X.shape}")

    for name, values in arrays.items():
        if len(values) != len(X):
            raise ValueError(
                f"Length mismatch: X has {len(X)} samples but "
                f"{name} has {len(values)} entries"
            )

    target_raw = signal_raw if task == "signal" else modulation_raw
    classes = sorted(np.unique(target_raw).tolist())
    snrs = sorted(float(value) for value in np.unique(snr_raw))

    if not classes:
        raise ValueError("No classes were found")

    if not snrs:
        raise ValueError("No SNR values were found")

    class_to_index = {name: index for index, name in enumerate(classes)}
    labels_integer = np.asarray(
        [class_to_index[name] for name in target_raw],
        dtype=np.int64,
    )

    lbl = [
        (str(target_raw[index]), float(snr_raw[index]))
        for index in range(len(target_raw))
    ]

    unique_sample_ids = np.unique(sample_id_raw)
    if len(unique_sample_ids) < 3:
        raise ValueError(
            "At least three unique sample_id values are required for "
            "train/validation/test splitting"
        )

    rng = np.random.default_rng(seed)
    shuffled_ids = rng.permutation(unique_sample_ids)

    n_ids = len(shuffled_ids)
    n_train_ids = int(np.floor(n_ids * train_ratio))
    n_val_ids = int(np.floor(n_ids * val_ratio))

    if n_train_ids == 0 or n_val_ids == 0:
        raise ValueError("The requested ratios produce an empty ID split")

    train_ids = shuffled_ids[:n_train_ids]
    val_ids = shuffled_ids[n_train_ids:n_train_ids + n_val_ids]
    test_ids = shuffled_ids[n_train_ids + n_val_ids:]

    if len(test_ids) == 0:
        raise ValueError("The requested ratios produce an empty test ID split")

    train_idx = np.flatnonzero(np.isin(sample_id_raw, train_ids))
    val_idx = np.flatnonzero(np.isin(sample_id_raw, val_ids))
    test_idx = np.flatnonzero(np.isin(sample_id_raw, test_ids))

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    split_indices = {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
    }
    split_ids = {
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids,
    }

    if set(train_ids) & set(val_ids):
        raise RuntimeError("sample_id leakage between train and validation")
    if set(train_ids) & set(test_ids):
        raise RuntimeError("sample_id leakage between train and test")
    if set(val_ids) & set(test_ids):
        raise RuntimeError("sample_id leakage between validation and test")

    all_indices = np.concatenate([train_idx, val_idx, test_idx])
    if len(all_indices) != len(X) or len(np.unique(all_indices)) != len(X):
        raise RuntimeError("The split does not cover every sample exactly once")

    expected_groups = {
        (class_id, snr)
        for class_id in range(len(classes))
        for snr in snrs
    }

    group_counts = {}
    for split_name, indices in split_indices.items():
        split_groups, counts = np.unique(
            np.stack(
                [labels_integer[indices], snr_raw[indices]],
                axis=1,
            ),
            axis=0,
            return_counts=True,
        )
        observed_groups = {
            (int(class_id), float(snr))
            for class_id, snr in split_groups
        }
        missing_groups = expected_groups - observed_groups

        if require_complete_grid and missing_groups:
            missing_names = [
                (classes[class_id], snr)
                for class_id, snr in sorted(missing_groups)
            ]
            raise ValueError(
                f"{split_name} is missing class-SNR groups: {missing_names}"
            )

        group_counts[split_name] = counts

    def materialize(indices):
        split_x = np.asarray(X[indices], dtype=np.float32)
        split_labels = labels_integer[indices]
        split_y = np.eye(len(classes), dtype=np.float32)[split_labels]
        return split_x, split_y

    X_train, Y_train = materialize(train_idx)
    X_val, Y_val = materialize(val_idx)
    X_test, Y_test = materialize(test_idx)

    Y_train_str = [classes[index] for index in labels_integer[train_idx]]

    print("=" * 80)
    print("RadCom grouped train/validation/test split")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"SNRs ({len(snrs)}): {snrs}")
    print(f"Total samples: {len(X)}")
    print(f"Unique sample IDs: {len(unique_sample_ids)}")

    for split_name in ("train", "validation", "test"):
        indices = split_indices[split_name]
        counts = group_counts[split_name]
        print(
            f"{split_name:>10}: samples={len(indices)}, "
            f"sample_ids={len(split_ids[split_name])}, "
            f"class-SNR count range={counts.min()}-{counts.max()}"
        )

    print("sample_id overlap: 0")
    print("=" * 80)

    return (
        (classes, snrs, lbl),
        (X_train, Y_train),
        (X_val, Y_val),
        (X_test, Y_test),
        (train_idx, val_idx, test_idx),
        Y_train_str,
    )


if __name__ == "__main__":
    (
        (classes, snrs, lbl),
        (X_train, Y_train),
        (X_val, Y_val),
        (X_test, Y_test),
        (train_idx, val_idx, test_idx),
        Y_train_str,
    ) = load_data_radcom_npy_grouped(
        data_dir=configured_path("RADIO_SSL_RADCOM_NPY", "data/radcom_npy"),
        seed=2016,
        task="signal",
        train_ratio=0.60,
        val_ratio=0.20,
        mmap_mode="r",
    )


    known_classes = [
        "Air-Ground-MTI",
        "Airborne-detection",
        "Airborne-range",
        "Ground mapping",
    ]

    unknown_classes = [
        "Radar Altimeter",
        "short-range",
    ]

    build_open_set_train_test(
        X_train=X_train,
        train_idx=train_idx,
        X_test=X_val,
        test_idx=val_idx,
        lbl=lbl,
        mods=classes,
        sequence_transform_func=sequence_transform,
        known_mods=known_classes,
        unknown_mods=unknown_classes,
        train_known_per_snr=10,
        test_per_snr=100,
        save_dir="./radar_open_set_5/validation",
        seed=2026,
    )

    build_open_set_train_test(
        X_train=X_train,
        train_idx=train_idx,
        X_test=X_test,
        test_idx=test_idx,
        lbl=lbl,
        mods=classes,
        sequence_transform_func=sequence_transform,
        known_mods=known_classes,
        unknown_mods=unknown_classes,
        train_known_per_snr=20,
        test_per_snr=100,
        save_dir="./radar_open_set_5/test",
        seed=2026,
    )

    save_dir = "./radar_open_set_5/test"


    split_open_test_by_snr(
        source_dir='./radar_open_set_5/test',
        output_dir="./radar_open_set_5/test/by_snr",
    )

    split_open_test_by_snr(
        source_dir='./radar_open_set_5/validation',
        output_dir="./radar_open_set_5/validation/by_snr",
    )
