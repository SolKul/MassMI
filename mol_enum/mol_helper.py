from rdkit import Chem


def smi_to_mol(sm_str, addHs=True):
    """SMILESの文字列をMolにする。そのときHをつけるかつけないか選ぶ。"""
    t_mol = Chem.MolFromSmiles(sm_str)
    if addHs:
        t_mol = Chem.AddHs(t_mol)
    return t_mol


def mol_to_smi_to_mol(mol, addHs=True):
    """molを一度SMILESになおして再度molにすることでエラーが出ないようにする"""
    smi = Chem.MolToSmiles(mol)
    n_mol = Chem.MolFromSmiles(smi)
    if addHs:
        n_mol = Chem.AddHs(n_mol)
    return n_mol
