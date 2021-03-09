"""
原子のリスト、またはmolに含まれる原子のリストから、分子を列挙する
"""

import numpy as np
import itertools
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image
import matplotlib.pyplot as plt

from . import mol_helper

def disp100mol(mol):
    im_mol=Draw.MolToImage(mol,size=(200,200))
    display(im_mol.resize((150,150),resample=5))
    
def Base_n_to_10(l_X,n):
    """
    n進数を10進数に直す。
    """
    out = 0
    for i in range(1,len(l_X)+1):
        out += int(l_X[-i])*(n**(i-1))
    return out#int out

def get_concat_h(l_im):
    width_sum=0
    for im in l_im:
        width_sum+=im.width
    dst = Image.new('RGB', (width_sum, l_im[0].height))
    width_p=0
    for im in l_im:
        dst.paste(im, (width_p, 0))
        width_p+=im.width
    return dst

def Conc_h_mols(l_mols):
    l_im=[]
    for mol in l_mols:
        im_mol=Draw.MolToImage(mol,size=(200,200))
        l_im.append(im_mol.resize((150,150),resample=5))
    if is_env_notebook():
        display(get_concat_h(l_im))
    else:
        im=get_concat_h(l_im)
        plt.imshow(im)
        plt.show()

def is_env_notebook():
    """Determine wheather is the environment Jupyter Notebook"""

    try:
        env_name = get_ipython().__class__.__name__
    except NameError:
        return False

    if env_name == 'TerminalInteractiveShell':
        # IPython shell
        return False
    # Jupyter Notebook (env_name == 'ZMQInteractiveShell')
    return True

bond_d={
    1:Chem.BondType.SINGLE,
    2:Chem.BondType.DOUBLE,
    3:Chem.BondType.TRIPLE
}

class Compound:
    """
    隣接行列からmolを生成するクラス

    Args:
        bond_mtx (:obj:`np.array`): 隣接行列
        atm_ind_list (list): Product内のアルゴリズムで、同じ分子に属していると判定された原子のインデックス
    """
    def __init__(
        self,
        atm_ind_list,
        atm_list,
        bond_mtx):
        self.atm_ind_list=atm_ind_list.copy()
        #もとの原子リストをコピー
        self.atm_list=atm_list.copy()
        #もとの隣接行列をコピー
        self.bond_mtx=bond_mtx

    def addatm(self,i):
        """
        i番目の原子を新しく化合物に追加
        """
        self.atm_ind_list.append(i)

    def exatms(self,ex_atm_ind_list):
        """
        ex_atm_ind_listで指定される原子を新しく化合物に追加
        """
        self.atm_ind_list.extend(ex_atm_ind_list)
    
    def GenMol(self):
        rwmol = Chem.RWMol()
        for i in self.atm_ind_list:
            # rwmolにAtomを追加
            rwmol.AddAtom(self.atm_list[i])
        for i in range(len(self.atm_ind_list)):
            # 対角要素はラジカルの数
            num_radical=self.bond_mtx[\
                self.atm_ind_list[i],\
                self.atm_ind_list[i]]
            if num_radical>=1:
                atom=rwmol.GetAtomWithIdx(i)
                atom.SetNumRadicalElectrons(int(num_radical))
            for j in range(i):
                # 結合を設定
                bond_num=self.bond_mtx[self.atm_ind_list[i],self.atm_ind_list[j]]
                if bond_num>=1:
                    rwmol.AddBond(i,j,bond_d[bond_num])
        self.cmol=mol_helper.mol_to_smi_to_mol(
            (Chem.AddHs(rwmol.GetMol())),
            addHs=True)

class Products:
    """
    隣接行列を化合物ごとに分離するクラス
    """
    def __init__(
        self,
        atm_list,
        bond_mtx,
        vlc_list):

        self.atm_list=atm_list
        self.bond_mtx=bond_mtx
        self.vlc_list=vlc_list
        self.generated=False

    def SplitComp(self):
        comp_c=0
        # どの原子がどの原子群に属しているかというリスト
        self.ar_c_t=(np.ones([len(self.atm_list)])*-1).astype('int')
        self.l_comp=[]
        for i in range(self.bond_mtx.shape[0]):
            #下三角行列のみ走査
            for j in range(i):
                if(self.bond_mtx[i,j]==0):
                    continue
                ar_b_c=self.ar_c_t[[i,j]]==[-1,-1]
                #原子が原子群に属しているかいないかのチェック
                if(np.all(ar_b_c)):
                    #i,jがどちらとも属していない場合
                    self.ar_c_t[[i,j]]=comp_c
                    self.l_comp.append(Compound([i,j],self.atm_list,self.bond_mtx))
                    comp_c+=1
                elif(ar_b_c[0]):
                    #iだけが所属していない場合
                    self.ar_c_t[i]=self.ar_c_t[j]
                    self.l_comp[self.ar_c_t[j]].addatm(i)
                elif(ar_b_c[1]):
                    #jだけが所属していない場合
                    self.ar_c_t[j]=self.ar_c_t[i]
                    self.l_comp[self.ar_c_t[i]].addatm(j)
                else:
                    if(self.ar_c_t[i]<self.ar_c_t[j]):
                        Comp=self.l_comp.pop(self.ar_c_t[j])
                        #繋がっていると分かった原子群同士のうち番号が大きい方をpopする
                        self.l_comp[self.ar_c_t[i]].exatms(Comp.l_atm)
                        #番号が小さい方に追加
                        for atm_num in Comp.l_atm:
                            self.ar_c_t[atm_num]=self.ar_c_t[i]
                            #繋がっていると分かった原子群のそれぞれの原子の属する原子群の番号を小さい方に統一
                    elif(self.ar_c_t[j]<self.ar_c_t[i]):
                        Comp=self.l_comp.pop(self.ar_c_t[i])
                        self.l_comp[self.ar_c_t[j]].exatms(Comp.l_atm)
                        for atm_num in Comp.l_atm:
                            self.ar_c_t[atm_num]=self.ar_c_t[j]
        for i in range(self.bond_mtx.shape[0]):
            if(self.ar_c_t[i]==-1):
                #原子群に分けた時点でどこにも所属していないものは原子
                self.ar_c_t[i]=comp_c
                self.l_comp.append(Compound([i],self.atm_list,self.bond_mtx))
                comp_c+=1

    def GenMols(self):
        #Molオブジェクトを生成
        self.SplitComp()
        l_mols=[]
        l_smiles=[]
        for Comp in self.l_comp:
            Comp.GenMol()
            l_mols.append(Comp.cmol)
            l_smiles.append(Chem.MolToSmiles(Comp.cmol))
        arg_s=np.argsort(l_smiles)
        self.mol_list=np.array(l_mols)[arg_s].tolist()
        self.smiles_list=sorted(l_smiles)
        self.str_smiles=''
        for smile in self.smiles_list:
            self.str_smiles+=' '+smile

        self.generated=True
        
    def Dispmols(self):
        if not self.generated:
            raise ValueError("not generated")
        Conc_h_mols(self.mol_list)
        print(self.str_smiles)

class CombProducts:
    """
    隣接行列を列挙し、生成物のリストを作るクラス
    """
    def __init__(self,b_exist_atm=True,b_exist_rad=True):
        self.b_exist_atm=b_exist_atm
        self.b_exist_rad=b_exist_rad
        self.generated=False

    def set_atm_list(self,str_atm_list):
        atm_list=[]
        for str_atm in str_atm_list:
            atm_list.append(Chem.AtomFromSmiles("["+str_atm+"]"))
        self.prepare_atm(atm_list)
    
    def set_mol(self,mol):
        self.prepare_atm(list(mol.GetAtoms()))

    def prepare_atm(self,atm_list):
        peri_tab=Chem.GetPeriodicTable()
        def default_val(atm):
            return peri_tab.GetDefaultValence(atm.GetAtomicNum())

        # 1.原子価、2.原子番号の優先順位で降順でソート
        self.atm_list=sorted(
            atm_list,
            key=lambda x: (default_val(x),x.GetAtomicNum()),
            reverse=True)

        # 原子番号と原子価のリストを用意
        self.atm_no_list=[]
        self.vlc_list=[]
        for atm in self.atm_list:
            self.atm_no_list.append(atm.GetAtomicNum())
            self.vlc_list.append(default_val(atm))

        self.num_atm=len(self.atm_list)

    def enumerate_bond_mtx(self):
        """
        重複組み合わせを用いて隣接行列を列挙する
        """
        l_bond_p=[[[0]*self.num_atm]]
        for k in range(1,self.num_atm):
            #重複組み合わせの玉の種類。
            #三角形の横方向の数+1

            #詳しくは足してnになる組み合わせの数の応用の
            #足してn以下になる組み合わせの数を参照
            conb_rep_kind=k+1
            # 結合する相手の原子の数への結合数が
            # 足して価電子数(valence_num)以下なるような場合の数
            # =注目している結合する相手の原子の数+1種類の玉から
            # 重複して選ぶときの場合の数
            valence_num=self.vlc_list[k]
            comb_rep_list=list(itertools.combinations_with_replacement(
                list(range(conb_rep_kind)),
                valence_num))
            l_bond_r_p=[]
            #隣接行列のある1行が取りうる行のリスト
            for i in range(len(comb_rep_list)):
                l_bond=[0]*self.num_atm
                for j in range(conb_rep_kind-1):
                    # 結合数=ある玉の数
                    b_num=comb_rep_list[i].count(j)
                    if(b_num>=4):
                        break
                    l_bond[j]=b_num
                else:
                    l_bond_r_p.append(l_bond)
            l_bond_p.append(l_bond_r_p)
        return l_bond_p

    def CalcComb(self):
        l_bond_p=self.enumerate_bond_mtx()

        ar_bond_p=np.array(list(itertools.product(*l_bond_p)))
        #隣接行列のある1行が取りうる行同士の直積
        for i in range(ar_bond_p.shape[0]):
            for j in range(ar_bond_p.shape[1]):
                ar_bond_p[i,:j,j]=ar_bond_p[i,j,:j]
                #対称行列にする
        ar_b_cons=ar_bond_p.sum(axis=1)<=np.array(self.vlc_list)
        #結合数が原子価以下のものをbooleanとして抽出
        if not self.b_exist_rad:
            #ラジカルの存在を許さない場合
            ar_b_cons=ar_bond_p.sum(axis=1)==np.array(self.vlc_list)
        ar_b_ind=np.all(ar_b_cons,axis=1)
        #すべての原子が、結合数が原子価以下であったらその隣接行列は整合性があると判断する
        #その整合性のあるもののindex
        if not self.b_exist_atm:
            #原子の存在を許さない場合
            ar_b_atm=np.all(ar_bond_p.sum(axis=1)!=0,axis=1)
            ar_b_ind=np.logical_and(ar_b_atm,ar_b_ind)
        ar_bond_cons=ar_bond_p[ar_b_ind]
        #整合性のあるもののみ取り出す。

        for i in range(ar_bond_cons.shape[0]):
            for j in range(ar_bond_cons.shape[1]):
                ar_bond_cons[i,j,j:]=0
                #下三角行列にする。

        l_l_base=[]
        for i in range(ar_bond_cons.shape[0]):
            l_base=[]
            for j in range(ar_bond_cons.shape[1]):
                #隣接行列の一つ一つの行を原子価+1進数とみなして十進数に直す。
                #この十進数にした数字で並び替えて、
                #同型のグラフを見つけ出す
                l_base.append(Base_n_to_10(ar_bond_cons[i,j,:],self.vlc_list[j]+1))
            l_l_base.append(l_base)
            #十進数に直した数字のリストをリストに追加
        ar_base=np.array(l_l_base)
        #このar_baseの1行について、原子種ごとにソートしたものは
        #同じグラフについては同じになる。(ただし同じにならない場合もある。)
        #ある一つのグラフに対して1意に決まるカノニカルラベルと言える
        #(厳密には違うが)
        
        atm_no_array=np.array(self.atm_no_list)
        #原子種のリストをarray化
        ar_uni=np.unique(self.atm_no_list)
        #原子のリストのユニークな要素を取り出し
        l_c_table=[]
        l_cons_b=[]
        for i in range(len(ar_base)):
            l_sorted=[]
            for atm_no in ar_uni:
                #原子リストのうち、注目している原子種と一致するところをbooleanで取り出す
                ar_uni_b=(atm_no_array==atm_no)
                #そのbooleanで取り出してsortし、原子種ごとにカノニカルラベルを作成
                #それをリストに追加し、グラフ全体のカノニカルラベルとする。
                l_sorted.extend(np.sort(ar_base[i][ar_uni_b]))
            if l_sorted not in l_c_table:
                #そのグラフ全体のカノニカルラベルがチェック表になければ、追加
                l_c_table.append(l_sorted)
                #カノニカルラベルがユニークなもののindexを追加
                l_cons_b.append(i)
        self.ar_bond_can=ar_bond_cons[l_cons_b]

        no_radical_bond_list=[]
        # 対称行列にする。
        for i in range(self.ar_bond_can.shape[0]):
            for j in range(self.ar_bond_can.shape[1]):
                self.ar_bond_can[i,:j,j]=self.ar_bond_can[i,j,:j]
                #i,i要素はラジカルなので原子価から結合数を引く
                num_radical=\
                    self.vlc_list[j]-\
                    self.ar_bond_can[i,:j,j].sum()-\
                    self.ar_bond_can[i,j+1:,j].sum()
                self.ar_bond_can[i,j,j]=num_radical
                # もしラジカルでなければ、以降の処理をスキップし、
                # 次のfor文(次の行)に移動
                if num_radical == 0:
                    continue
                for k in range(j):
                    # 注目原子(k)がラジカルでなければ、以降の処理をスキップし、
                    # 次の原子に移動
                    if not self.ar_bond_can[i,j,k]>0:
                        continue
                    # 行の原子(j)と注目原子(k)の結合していなければ
                    if self.ar_bond_can[i,k,k] == 0:
                        continue
                    # 上のすべての条件を満たした場合、ラジカル同士の結合があると判断
                    # beakしelseは飛ばす。
                    break
                else:
                    continue
                # ラジカル同士の結合がある場合ここに飛ぶ
                # beakしelseは飛ばす。
                break
            else:
                # ラジカル同士の結合がないと判断されれば
                # リストに追加
                no_radical_bond_list.append(i)
        self.ar_bond_can=self.ar_bond_can[no_radical_bond_list]
                    
    def GenProComb(self):
        self.CalcComb()
        l_l_smiles=[]
        self.l_prod=[]
        self.comb_c=0
        self.smiles_set=set()
        for i in range(self.ar_bond_can.shape[0]):
            p_t=Products(
                self.atm_list,
                self.ar_bond_can[i],
                self.vlc_list)
            p_t.GenMols()
            if p_t.str_smiles in l_l_smiles:
                #smilesのリストにあれば飛ばす
                continue
            l_l_smiles.append(p_t.str_smiles)
            self.l_prod.append(p_t)
            self.smiles_set=self.smiles_set.union(p_t.smiles_list)
            self.comb_c+=1
        self.generated=True

    def DispComb(self):
        if not self.generated:
            raise ValueError("not generated")

        for i in range(self.comb_c):
            print('組み合わせ:'+str(i))
            self.l_prod[i].Dispmols()
            print()