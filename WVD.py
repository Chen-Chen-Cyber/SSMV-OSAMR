from project_paths import configured_path
import numpy as np
from scipy.signal import get_window
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from self import data_transform_masked4cl
from random import *
import pywt
import torch.nn.functional as F
from Data_Augmentation import Data_Augmentation


def pwvd(x, window_type="hamming", win_len=129):
    """
    Smoothed Wigner-Ville Distribution (PWVD)
    x : 1D numpy array (complex)
    window_type: e.g. 'hamming', 'hann'
    """

    N = len(x)
    wvd_matrix = np.zeros((N, N), dtype=np.complex64)

    # analysis window
    window = get_window(window_type, win_len)
    half = win_len // 2

    for t in range(N):
        for tau in range(-half, half + 1):
            t_pos = t + tau
            t_neg = t - tau
            if 0 <= t_pos < N and 0 <= t_neg < N:
                w = window[tau + half]
                wvd_matrix[t, tau + half] = w * x[t_pos] * np.conj(x[t_neg])

    # FFT along tau to get frequency axis
    wvd_tf = np.fft.fftshift(np.fft.fft(wvd_matrix, axis=1), axes=1)
    return np.abs(wvd_tf)

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
            X.append(Xd[(mod, snr)])
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

if __name__ == "__main__":
    (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx),Y_train_str =\
            load_data()

def _8PSK():
    filename = configured_path("RADIO_SSL_RML2016A", "data/raw/RML2016.10a_dict.pkl")
    with open(filename, 'rb') as file:
        Xd = pickle.load(file, encoding='bytes')

    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X1 = []
    Y1 = []
    lbl = []
    str = b'CPFSK'
    for mod in mods:
        for snr in snrs:
            if (mod == str and snr == -10):
                test = Xd[(mod, snr)]
                X1.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
                for i in range(Xd[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
                    Y1.append([test[0] + 1j * test[1]])
    X1 = np.vstack(X1)


    np.save("./CPFSK_-10dB.npy",X1)


import os
from pathlib import Path

import numpy as np


def sequence_transform_chunked(
    data,
    transform_func,
    chunk_size=2048,
):
    'Apply sequence_transform in chunks to reduce peak memory. Input: (N, 2, 128). Output: (N, 10, 128).'
    data = np.asarray(data, dtype=np.float32)

    if len(data) == 0:
        return np.empty(
            (0, 10, data.shape[-1]),
            dtype=np.float32,
        )

    first_end = min(chunk_size, len(data))

    first_output = transform_func(
        data[:first_end]
    ).astype(np.float32)

    output_shape = (
        len(data),
        *first_output.shape[1:],
    )

    sequence_data = np.empty(
        output_shape,
        dtype=np.float32,
    )

    sequence_data[:first_end] = first_output

    for start in range(first_end, len(data), chunk_size):
        end = min(start + chunk_size, len(data))

        sequence_data[start:end] = transform_func(
            data[start:end]
        ).astype(np.float32)

    return sequence_data


def build_open_set_train_test(
    X_train,
    train_idx,
    X_test,
    test_idx,
    lbl,
    mods,
    sequence_transform_func,
    known_mods=None,
    unknown_mods=None,
    train_known_per_snr=200,
    test_per_snr=100,
    save_dir="./open_set_train_test_WBFM",
    seed=2026,
    transform_chunk_size=2048,
):
    'Build open-set training and test sets. Training contains known classes only, with 200 samples per modulation and SNR. Testing contains known and unknown classes, with 100 samples per modulation and SNR. Map all unknown modulations to one Unknown label. Save raw two-channel IQ, ten-channel features, integer labels, SNRs, unknown flags and original global indices. Counts are controlled by the function arguments.'

    if known_mods is None:
        known_mods = [
            "BPSK",
            "QPSK",
            "8PSK",
            "QAM16",
            "QAM64",
            "PAM4",
            "AM-DSB",
            "WBFM",
            "GFSK",
        ]

    if unknown_mods is None:
        unknown_mods = [
            "AM-SSB",
            "CPFSK",
        ]

    known_mods = list(known_mods)
    unknown_mods = list(unknown_mods)
    mods = list(mods)


    overlap = set(known_mods) & set(unknown_mods)

    if overlap:
        raise ValueError(
            f"Known and unknown classes overlap: {overlap}"
        )

    missing_mods = (
        set(known_mods + unknown_mods)
        - set(mods)
    )

    if missing_mods:
        raise ValueError(
            f"Classes not found in the dataset: {missing_mods}"
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    train_idx = np.asarray(
        train_idx,
        dtype=np.int64,
    )

    test_idx = np.asarray(
        test_idx,
        dtype=np.int64,
    )

    rng = np.random.default_rng(seed)

    known_to_new = {
        mod: class_index
        for class_index, mod in enumerate(known_mods)
    }

    unknown_label = len(known_mods)

    open_set_classes = (
        known_mods + ["Unknown"]
    )


    snr_min = -4
    snr_max = 4

    snrs = sorted({
        float(item[1])
        for item in lbl
        if snr_min <= float(item[1]) <= snr_max
    })

    if not snrs:
        raise ValueError(
            f'No samples have SNRs in [{snr_min}, {snr_max}] dB'
        )

    print(f'SNRs used for this split: {snrs}')


    train_mod_names = np.asarray(
        [
            lbl[int(global_index)][0]
            for global_index in train_idx
        ],
        dtype=object,
    )

    train_snr_values = np.asarray(
        [
            lbl[int(global_index)][1]
            for global_index in train_idx
        ],
        dtype=np.int32,
    )

    test_mod_names = np.asarray(
        [
            lbl[int(global_index)][0]
            for global_index in test_idx
        ],
        dtype=object,
    )

    test_snr_values = np.asarray(
        [
            lbl[int(global_index)][1]
            for global_index in test_idx
        ],
        dtype=np.int32,
    )


    selected_train_positions = []

    for mod in known_mods:
        for snr in snrs:
            positions = np.flatnonzero(
                (train_mod_names == mod)
                & (train_snr_values == snr)
            )

            if len(positions) < train_known_per_snr:
                raise ValueError(
                    f"Training group ({mod}, {snr} dB) "
                    f"contains only {len(positions)} samples; "
                    f"{train_known_per_snr} are required."
                )

            selected = rng.choice(
                positions,
                size=train_known_per_snr,
                replace=False,
            )

            selected_train_positions.extend(
                selected.tolist()
            )

    selected_train_positions = np.asarray(
        selected_train_positions,
        dtype=np.int64,
    )


    selected_train_positions = rng.permutation(
        selected_train_positions
    )

    train_raw_2ch = X_train[
        selected_train_positions
    ].astype(np.float32)

    train_labels = np.asarray(
        [
            known_to_new[mod]
            for mod in train_mod_names[
                selected_train_positions
            ]
        ],
        dtype=np.int64,
    )

    train_snrs = train_snr_values[
        selected_train_positions
    ].astype(np.int32)

    train_source_indices = train_idx[
        selected_train_positions
    ].astype(np.int64)

    print("Transforming known training data...")

    train_sequence_10ch = sequence_transform_chunked(
        train_raw_2ch,
        transform_func=sequence_transform_func,
        chunk_size=transform_chunk_size,
    )


    selected_test_positions = []

    for mod in known_mods + unknown_mods:
        for snr in snrs:
            positions = np.flatnonzero(
                (test_mod_names == mod)
                & (test_snr_values == snr)
            )

            if len(positions) < test_per_snr:
                raise ValueError(
                    f"Test group ({mod}, {snr} dB) "
                    f"contains only {len(positions)} samples; "
                    f"{test_per_snr} are required."
                )

            selected = rng.choice(
                positions,
                size=test_per_snr,
                replace=False,
            )

            selected_test_positions.extend(
                selected.tolist()
            )

    selected_test_positions = np.asarray(
        selected_test_positions,
        dtype=np.int64,
    )


    selected_test_positions = rng.permutation(
        selected_test_positions
    )

    test_raw_2ch = X_test[
        selected_test_positions
    ].astype(np.float32)

    selected_test_mods = test_mod_names[
        selected_test_positions
    ]

    test_labels = np.asarray(
        [
            (
                known_to_new[mod]
                if mod in known_to_new
                else unknown_label
            )
            for mod in selected_test_mods
        ],
        dtype=np.int64,
    )

    test_snrs = test_snr_values[
        selected_test_positions
    ].astype(np.int32)

    test_is_unknown = np.asarray(
        [
            mod in unknown_mods
            for mod in selected_test_mods
        ],
        dtype=np.bool_,
    )

    test_source_indices = test_idx[
        selected_test_positions
    ].astype(np.int64)

    print("Transforming open-set test data...")

    test_sequence_10ch = sequence_transform_chunked(
        test_raw_2ch,
        transform_func=sequence_transform_func,
        chunk_size=transform_chunk_size,
    )


    if not (
        len(train_raw_2ch)
        == len(train_sequence_10ch)
        == len(train_labels)
        == len(train_snrs)
    ):
        raise RuntimeError(
            "Training data dimensions are inconsistent."
        )

    if not (
        len(test_raw_2ch)
        == len(test_sequence_10ch)
        == len(test_labels)
        == len(test_snrs)
        == len(test_is_unknown)
    ):
        raise RuntimeError(
            "Test data dimensions are inconsistent."
        )


    for mod_index, mod in enumerate(known_mods):
        for snr in snrs:
            count = np.sum(
                (train_labels == mod_index)
                & (train_snrs == snr)
            )

            if count != train_known_per_snr:
                raise RuntimeError(
                    f"Incorrect training count for "
                    f"({mod}, {snr} dB): {count}"
                )


    for mod in known_mods + unknown_mods:
        for snr in snrs:
            count = np.sum(
                (selected_test_mods == mod)
                & (test_snrs == snr)
            )

            if count != test_per_snr:
                raise RuntimeError(
                    f"Incorrect test count for "
                    f"({mod}, {snr} dB): {count}"
                )


    np.save(
        save_dir / "train_known_raw_2ch_data.npy",
        train_raw_2ch,
    )

    np.save(
        save_dir / "train_known_sequence_10ch_data.npy",
        train_sequence_10ch,
    )

    np.save(
        save_dir / "train_known_label.npy",
        train_labels,
    )

    np.save(
        save_dir / "train_known_snr.npy",
        train_snrs,
    )

    np.save(
        save_dir / "train_known_source_index.npy",
        train_source_indices,
    )


    np.save(
        save_dir / "test_open_raw_2ch_data.npy",
        test_raw_2ch,
    )

    np.save(
        save_dir / "test_open_sequence_10ch_data.npy",
        test_sequence_10ch,
    )

    np.save(
        save_dir / "test_open_label.npy",
        test_labels,
    )

    np.save(
        save_dir / "test_open_snr.npy",
        test_snrs,
    )

    np.save(
        save_dir / "test_open_is_unknown.npy",
        test_is_unknown,
    )

    np.save(
        save_dir / "test_open_original_mod.npy",
        selected_test_mods,
    )

    np.save(
        save_dir / "test_open_source_index.npy",
        test_source_indices,
    )


    np.save(
        save_dir / "known_mods.npy",
        np.asarray(known_mods),
    )

    np.save(
        save_dir / "unknown_mods.npy",
        np.asarray(unknown_mods),
    )

    np.save(
        save_dir / "open_set_classes.npy",
        np.asarray(open_set_classes),
    )

    np.save(
        save_dir / "snrs.npy",
        np.asarray(snrs, dtype=np.int32),
    )


    print("=" * 70)
    print("Open-set dataset construction completed.")
    print(f"Save directory: {save_dir.resolve()}")

    print("\nKnown training set")
    print("2-channel data:", train_raw_2ch.shape)
    print("10-channel data:", train_sequence_10ch.shape)
    print("Labels:", train_labels.shape)
    print("Unique labels:", np.unique(train_labels))
    print("SNRs:", np.unique(train_snrs))

    print("\nOpen-set test set")
    print("2-channel data:", test_raw_2ch.shape)
    print("10-channel data:", test_sequence_10ch.shape)
    print("Labels:", test_labels.shape)
    print("Unique labels:", np.unique(test_labels))
    print("Known samples:", np.sum(~test_is_unknown))
    print("Unknown samples:", np.sum(test_is_unknown))
    print("SNRs:", np.unique(test_snrs))

    print("\nClass mapping")
    for class_index, class_name in enumerate(
        open_set_classes
    ):
        print(f"{class_index}: {class_name}")

    return {
        "train_raw_2ch": train_raw_2ch,
        "train_sequence_10ch": train_sequence_10ch,
        "train_labels": train_labels,
        "train_snrs": train_snrs,
        "test_raw_2ch": test_raw_2ch,
        "test_sequence_10ch": test_sequence_10ch,
        "test_labels": test_labels,
        "test_snrs": test_snrs,
        "test_is_unknown": test_is_unknown,
        "known_mods": known_mods,
        "unknown_mods": unknown_mods,
        "open_set_classes": open_set_classes,
    }

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

from pathlib import Path
import numpy as np


def format_snr_name(snr):
    'Convert an SNR to a filename component: -20 -> m20, 0 -> p0, 18 -> p18.'
    snr = int(snr)

    if snr < 0:
        return f"m{abs(snr)}"

    return f"p{snr}"

from pathlib import Path
import numpy as np


def format_snr_name(snr):
    return str(int(snr))


def split_open_test_by_snr(
    source_dir="./open_set_train_test/WBFM",
    output_dir=None,
):
    'Split open-set test files produced by build_open_set_train_test into SNR-specific directories. Save raw two-channel IQ, ten-channel features, integer and one-hot labels, known/unknown flags, original modulation names, original sample indices and SNR labels.'
    source_dir = Path(source_dir)

    if output_dir is None:
        output_dir = source_dir / "test_by_snr"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    required_files = {
        "raw": source_dir / "test_open_raw_2ch_data.npy",
        "sequence": source_dir / "test_open_sequence_10ch_data.npy",
        "label": source_dir / "test_open_label.npy",
        "snr": source_dir / "test_open_snr.npy",
        "unknown": source_dir / "test_open_is_unknown.npy",
        "original_mod": source_dir / "test_open_original_mod.npy",
        "source_index": source_dir / "test_open_source_index.npy",
        "classes": source_dir / "open_set_classes.npy",
    }

    missing_files = [
        str(path)
        for path in required_files.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            'Required files are missing. Run build_open_set_train_test first:\n'
            + "\n".join(missing_files)
        )


    test_raw_2ch = np.load(
        required_files["raw"],
        allow_pickle=False
    )

    test_sequence_10ch = np.load(
        required_files["sequence"],
        allow_pickle=False
    )

    test_labels = np.load(
        required_files["label"],
        allow_pickle=False
    ).astype(np.int64)

    test_snrs = np.load(
        required_files["snr"],
        allow_pickle=False
    ).astype(np.int32)

    test_is_unknown = np.load(
        required_files["unknown"],
        allow_pickle=False
    ).astype(np.bool_)


    test_original_mods = np.load(
        required_files["original_mod"],
        allow_pickle=True
    )

    test_source_indices = np.load(
        required_files["source_index"],
        allow_pickle=False
    ).astype(np.int64)

    open_set_classes = np.load(
        required_files["classes"],
        allow_pickle=True
    )


    if test_raw_2ch.ndim != 3:
        raise ValueError(
            f'Invalid two-channel data dimensions: {test_raw_2ch.shape}'
        )

    if test_sequence_10ch.ndim != 3:
        raise ValueError(
            f'Invalid ten-channel data dimensions: {test_sequence_10ch.shape}'
        )

    if test_raw_2ch.shape[1] != 2:
        raise ValueError(
            f'Raw data must have two channels; received shape: {test_raw_2ch.shape}'
        )

    if test_sequence_10ch.shape[1] != 10:
        raise ValueError(
            f'Transformed data must have ten channels; received shape: {test_sequence_10ch.shape}'
        )

    lengths = [
        len(test_raw_2ch),
        len(test_sequence_10ch),
        len(test_labels),
        len(test_snrs),
        len(test_is_unknown),
        len(test_original_mods),
        len(test_source_indices),
    ]

    if len(set(lengths)) != 1:
        raise ValueError(
            f'Test data and label counts differ: {lengths}'
        )

    num_classes = len(open_set_classes)

    if test_labels.min() < 0:
        raise ValueError('Labels contain negative values.')

    if test_labels.max() >= num_classes:
        raise ValueError(
            f'Maximum label is {test_labels.max()}, but the total number of classes is {num_classes}。'
        )


    unique_snrs = np.sort(
        np.unique(test_snrs)
    )

    np.save(
        output_dir / "snrs.npy",
        unique_snrs.astype(np.int32)
    )

    np.save(
        output_dir / "open_set_classes.npy",
        open_set_classes
    )

    total_saved = 0

    for snr in unique_snrs:
        snr = int(snr)
        snr_name = format_snr_name(snr)

        positions = np.flatnonzero(
            test_snrs == snr
        )

        raw_snr = test_raw_2ch[positions].astype(
            np.float32
        )

        sequence_snr = test_sequence_10ch[
            positions
        ].astype(np.float32)

        labels_snr = test_labels[positions]
        unknown_snr = test_is_unknown[positions]

        original_mods_snr = test_original_mods[
            positions
        ]

        source_indices_snr = test_source_indices[
            positions
        ]

        snr_values = test_snrs[positions]


        labels_onehot_snr = np.zeros(
            (len(labels_snr), num_classes),
            dtype=np.float32
        )

        labels_onehot_snr[
            np.arange(len(labels_snr)),
            labels_snr
        ] = 1.0

        snr_dir = (
            output_dir / f"snr_{snr_name}dB"
        )

        snr_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            snr_dir / (
                f"raw_test_{snr_name}dB_data.npy"
            ),
            raw_snr
        )

        np.save(
            snr_dir / (
                f"sequence_test_{snr_name}dB_data.npy"
            ),
            sequence_snr
        )

        np.save(
            snr_dir / (
                f"test_{snr_name}dB_label.npy"
            ),
            labels_snr
        )

        np.save(
            snr_dir / (
                f"test_{snr_name}dB_label_onehot.npy"
            ),
            labels_onehot_snr
        )

        np.save(
            snr_dir / (
                f"test_{snr_name}dB_is_unknown.npy"
            ),
            unknown_snr
        )

        np.save(
            snr_dir / (
                f"test_{snr_name}dB_original_mod.npy"
            ),
            original_mods_snr
        )

        np.save(
            snr_dir / (
                f"test_{snr_name}dB_source_index.npy"
            ),
            source_indices_snr
        )

        np.save(
            snr_dir / (
                f"test_{snr_name}dB_snr.npy"
            ),
            snr_values
        )

        known_count = int(
            np.sum(~unknown_snr)
        )

        unknown_count = int(
            np.sum(unknown_snr)
        )

        total_saved += len(positions)

        print(
            f"SNR={snr:>3} dB | "
            f"total={len(positions)} | "
            f"known={known_count} | "
            f"unknown={unknown_count} | "
            f"raw={raw_snr.shape} | "
            f"sequence={sequence_snr.shape}"
        )


    if total_saved != len(test_labels):
        raise RuntimeError(
            f'Incorrect saved sample count: {total_saved} != {len(test_labels)}'
        )

    print("=" * 70)
    print('Finished splitting the test set by SNR.')
    print(f'Input directory: {source_dir.resolve()}')
    print(f'Output directory: {output_dir.resolve()}')
    print(f'Total samples: {total_saved}')
    print(f'SNR list: {unique_snrs.tolist()}')

    return output_dir
