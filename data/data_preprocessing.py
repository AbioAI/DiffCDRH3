import numpy as np


def read_sequences(cdrh3_file, epitope_file):
    """ 读取 CDRH3 和 Epitope 序列 """
    with open(cdrh3_file, 'r') as f:
        cdrh3_list = [line.strip() for line in f if line.strip()]

    with open(epitope_file, 'r') as f:
        epitope_list = [line.strip() for line in f if line.strip()]

    return cdrh3_list, epitope_list


def onehot_encode(sequences, amino_acids, aa_dict):
    """ 对蛋白质序列进行 One-hot 编码 """
    encoded_sequences = []
    for seq in sequences:
        onehot = np.zeros((len(seq), len(amino_acids)), dtype=np.float32)
        for i, aa in enumerate(seq):
            if aa in aa_dict:
                onehot[i, aa_dict[aa]] = 1
        encoded_sequences.append(onehot)

    return np.array(encoded_sequences, dtype=np.float32)


def save_onehot(cdrh3_npy_file, epitope_npy_file):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_dict = {aa: idx for idx, aa in enumerate(amino_acids)}
    cdrh3_sequences, epitope_sequences = read_sequences("CDRH3.txt", "Epitope.txt")

    # 进行 One-hot 编码
    cdrh3_onehot = onehot_encode(cdrh3_sequences, amino_acids, aa_dict)
    epitope_onehot = onehot_encode(epitope_sequences, amino_acids, aa_dict)

    np.save(cdrh3_npy_file, cdrh3_onehot)
    np.save(epitope_npy_file, epitope_onehot)

    print(f"Saved successfully!")
