def replace_H_with_Li(mol):
    """HをLiに入れ替える"""
    mol=Chem.AddHs(mol)
    rw_mol=Chem.RWMol(mol)
    for atm in rw_mol.GetAtoms():
        if atm.GetSymbol() != "H":
            if atm.GetSymbol() == "Li":
                raise ValueError("もともとLiが含まれている")
            continue
        idx=atm.GetIdx()
        rw_mol.ReplaceAtom(idx,atm_Li)
    Chem.SanitizeMol(rw_mol)
    return rw_mol.GetMol()

def replace_rad_with_Na(mol):
    """radをNaに入れ替える"""
    rw_mol=Chem.RWMol(mol)
    atm_num=rw_mol.GetNumAtoms()
    na_idx=atm_num
    
    for idx in range(atm_num):
        atm=rw_mol.GetAtomWithIdx(idx)
        if atm.GetSymbol() == "Na":
            raise ValueError("もともとNaが含まれている")
        rad_num=atm.GetNumRadicalElectrons()
        if rad_num==0:
            continue
        atm.SetNumRadicalElectrons(0)
        for j in range(rad_num):
            # Naを加える
            rw_mol.AddAtom(atm_Na) 
            # ラジカルのあった原子とNaをつなぐ
            rw_mol.AddBond(idx,na_idx,order=Chem.BondType.SINGLE)
            na_idx +=1
            
    Chem.SanitizeMol(rw_mol)
    return rw_mol.GetMol()

def replace_rad_with_H(mol):
    """radをHに入れ替える"""
    mol=Chem.AddHs(mol)
    rw_mol=Chem.RWMol(mol)
    for atm in rw_mol.GetAtoms():
        if atm.GetSymbol() == "H":
            raise ValueError("もともとHが含まれている")
        rad_num=atm.GetNumRadicalElectrons()
        atm.SetNumRadicalElectrons(0)
        atm.SetNumExplicitHs(rad_num)
    Chem.SanitizeMol(rw_mol)
    e_mol=rw_mol.GetMol()
    return e_mol