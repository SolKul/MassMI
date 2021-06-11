import numpy as np
import itertools
from rdkit import Chem
from graphillion import GraphSet

# 原子番号→色のリスト
color_dict = {
    1: "silver",
    6: "black",
    9: "deepskyblue",
    17: "lawngreen"
}
def atm_color(atomic_no):
    """
    原子番号→色を返す
    """
    if atomic_no in color_dict:
        color=color_dict[atomic_no]
    else:
        color="brown"
    return color

#周期表オブジェクト
peri_tab = Chem.GetPeriodicTable()


def default_vlc(atm):
    """
    デフォルトの原子価を返す
    """
    return peri_tab.GetDefaultValence(atm.GetAtomicNum())


def Base_n_to_10(l_X, n):
    """
    n進数を10進数に直す。
    """
    out = 0
    for i in range(1, len(l_X)+1):
        out += int(l_X[-i])*(n**(i-1))
    return out  # int out

"""
結合の辞書
"""
bond_d = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE
}

class SpanMolEnum:
    """
    与えられた原子すべてを使う全域グラフを列挙するクラス
    """

    def __init__(
            self,
            b_exist_rad=True,
            max_bias_rad_num=1,
            max_bias_bonds=1):
        self.b_exist_rad = b_exist_rad
        self.max_bias_rad_num=max_bias_rad_num
        self.max_bias_bonds=max_bias_bonds
        self.generated = False

    def set_atm_list(self, str_atm_list):
        atm_list = []
        for str_atm in str_atm_list:
            atm_list.append(Chem.AtomFromSmiles("["+str_atm+"]"))
        self.prepare_atm(atm_list)

    def set_mol(self, mol):
        self.prepare_atm(list(mol.GetAtoms()))

    def prepare_atm(self, atm_list):

        # 1.原子価、2.原子番号の優先順位で降順でソート
        self.atm_list = sorted(
            atm_list,
            key=lambda x: (default_vlc(x), -x.GetAtomicNum()),
            reverse=True)

        # 原子価のリスト
        self.atm_no_list = []
        self.atomic_no_list = []
        self.vlc_list = []
        self.degree_constraints = dict()
        self.atm_colors_list = []
        for i, atm in enumerate(self.atm_list):
            self.atm_no_list.append(i)
            self.atomic_no_list.append(atm.GetAtomicNum())

            # 次数の制約を決定する
            vlc_num = default_vlc(atm)
            if vlc_num > 1:
                # 結合次数を1~原子価にすることで二重結合も発生させる
                self.degree_constraints[i] = range(1, vlc_num+1)
            else:
                self.degree_constraints[i] = 1

            self.vlc_list.append(vlc_num)

            # atmから原子番号を取り出し、原子番号→色の辞書で色を取り出し
            # リストに追加
            self.atm_colors_list.append(atm_color(atm.GetAtomicNum()))

        self.atm_num = len(self.atm_list)

        # 全体グラフ(有りうる結合の集合)を作成
        self.bond_universe = []
        # 原子の集まりから2つを選ぶ組み合わせ(nC2)
        for atm_comb in itertools.combinations(self.atm_no_list, 2):
            def_vlc_1 = default_vlc(
                self.atm_list[atm_comb[0]])
            def_vlc_2 = default_vlc(
                self.atm_list[atm_comb[1]])
            # どちらの原子価も1であれば、そこの結合は全体グラフには入れない。
            # ただし原子数が2の場合を除く
            # つまり、(原子数が2)or(原子価1が1でない)or(原子価2が1でない)のどれかを満たすとき、
            # リストに加える。
            if (self.atm_num==2) or (def_vlc_1 != 1) or (def_vlc_2 != 1):
                self.bond_universe.append(atm_comb)

    def generate_mol(self):
        if self.atm_num == 1:
            if self.b_exist_rad:
                str_atm=self.atm_list[0].GetSymbol()
                self.smi_set={"["+str_atm+"]"}
            else:
                self.smi_set=set()
        elif len(self.bond_universe)==0:
            self.smi_set=set()
        else:
            self.enumerate()
            self.narrow_cands() 
        self.generated=True

    def enumerate(self):
        GraphSet.set_universe(self.bond_universe)
        # 全域グラフのみ列挙
        self.paths = GraphSet.graphs(
            vertex_groups=[self.atm_no_list],
            degree_constraints=self.degree_constraints,
            no_loop=True)
        self.paths_num = len(self.paths)

    def narrow_cands(self):

        bond_mtx = np.zeros(
            [self.paths_num, self.atm_num, self.atm_num], dtype=int)

        path_no = 0
        # 一つの候補を取り出し
        for path in self.paths:
            # 結合行列(グラフ理論での隣接行列を作成)
            for bond in path:
                bond_mtx[path_no, bond[0], bond[1]] = 1
                bond_mtx[path_no, bond[1], bond[0]] = 1
            path_no += 1

        ar_base = np.zeros(
            [self.paths_num, self.atm_num], dtype=int)
        for i in range(self.paths_num):
            for j in range(1, self.atm_num):
                # 隣接行列下三角行列の一つ一つの行を原子価+1進数とみなして十進数に直す。
                ar_base[i, j] = Base_n_to_10(
                    bond_mtx[i, j, :j],
                    self.vlc_list[j]+1)
        # このar_baseの1行について、原子種ごとにソートしたものは
        # 同じグラフについては同じになる。(ただし同じにならない場合もある。)
        # ある一つのグラフに対して1意に決まるカノニカルラベルと言える
        # (厳密には違うが)
        # これ使い同型のグラフを見つけ出す

        # 原子番号のリストをarray化
        atomic_no_array = np.array(self.atomic_no_list)
        # 原子のリストのユニークな要素を取り出し
        ar_uni = np.unique(self.atomic_no_list)
        uni_atm_num = len(ar_uni)

        # 原子リストのうち、注目している原子種と一致するところをbooleanで取り出す
        aten_atm_bool = []
        for atm_no in ar_uni:
            aten_atm_bool.append(atomic_no_array == atm_no)

        l_c_table = []
        uni_bool = [False]*self.paths_num
        for i in range(self.paths_num):
            l_sorted = []
            for j in range(uni_atm_num):
                # booleanで取り出してsortし、原子種ごとにカノニカルラベルを作成
                # それをリストに追加し、グラフ全体のカノニカルラベルとする。
                l_sorted.extend(np.sort(ar_base[i][aten_atm_bool[j]]))
            if l_sorted not in l_c_table:
                # そのグラフ全体のカノニカルラベルがチェック表になければ、追加
                l_c_table.append(l_sorted)
                # カノニカルラベルがユニークなもののindexを追加
                uni_bool[i]=True
        n_bond_mtx = bond_mtx[uni_bool, :, :]
        cand_num=round(np.sum(uni_bool))

        # 一つの候補を取り出し
        for ind in range(cand_num):
            for i in range(self.atm_num):
                rad_num = self.vlc_list[i]-np.sum(n_bond_mtx[ind, i, :])
                n_bond_mtx[ind, i, i] = rad_num
            for i in range(self.atm_num):
                # ラジカルでない場合
                if n_bond_mtx[ind, i, i] == 0:
                    continue
                # ある別の原子とのつながりを見る
                for j in range(i):
                    if n_bond_mtx[ind, i, j] == 0:
                        continue
                    if n_bond_mtx[ind, j, j] == 0:
                        continue
                    # ラジカル同士の結合がある場合、
                    # 双方のラジカル数最小値を取り出し、
                    # 双方のラジカル数から引くとともに、
                    # 結合数に加える
                    min_rad_num = min(
                        n_bond_mtx[ind, i, i], n_bond_mtx[ind, j, j])
                    n_bond_mtx[ind, i, i] -= min_rad_num
                    n_bond_mtx[ind, j, j] -= min_rad_num
                    n_bond_mtx[ind, i, j] += min_rad_num
                    n_bond_mtx[ind, j, i] += min_rad_num

        # ラジカルが許されなければ
        if not self.b_exist_rad:
            no_rad_bool=[False]*cand_num
            for ind in range(cand_num):
                    for i in range(self.atm_num):
                        #隣接行列の対角成分が0以上なら
                        if n_bond_mtx[ind, i, i] > 0:
                            break
                    else:
                        no_rad_bool[ind] = True
            n_bond_mtx=n_bond_mtx[no_rad_bool,:,:]
            cand_num=round(np.sum(no_rad_bool))

        # 原子価4の原子の数
        vlc_mt_4=self.atm_num
        for i,vlc_num in enumerate(self.vlc_list):
            if vlc_num<4:
                vlc_mt_4=i
                break

        # 原子価4の原子の数が2以下なら
        if vlc_mt_4 <3:
            no_bias_bool=self.bias_check_u3(
                n_bond_mtx,
                cand_num,
                vlc_mt_4)
        else:
            no_bias_bool=self.bias_check_o3(
                n_bond_mtx,
                cand_num,
                vlc_mt_4)

        n_bond_mtx=n_bond_mtx[no_bias_bool,:,:]
        narrow_num=round(np.sum(no_bias_bool))

        self.smi_set = set()

        for ind in range(narrow_num):
            rwmol = Chem.RWMol()
            for atm in self.atm_list:
                # rwmolにAtomを追加
                rwmol.AddAtom(atm)
            for i in range(self.atm_num):
                # 対角要素はラジカルの数
                num_radical = n_bond_mtx[ind, i, i]
                if num_radical >= 1:
                    atom = rwmol.GetAtomWithIdx(i)
                    atom.SetNumRadicalElectrons(int(num_radical))
                for j in range(i):
                    # 結合を設定
                    bond_num = n_bond_mtx[ind, i, j]
                    if bond_num >= 1:
                        mol = rwmol.AddBond(i, j, bond_d[bond_num])
            smi = Chem.MolToSmiles(rwmol)
            self.smi_set.add(smi)

    def bias_check_u3(
            self,
            n_bond_mtx,
            cand_num,
            vlc_mt_4):
        """
        原子価4の原子の数が2以下の場合、
        C同士が繋がっていて、ラジカルの差が指定以上か、
        Cしかない化合物の場合を除外
        """
        no_bias_bool=[False]*cand_num
        for ind in range(cand_num):
            for i in range(vlc_mt_4):
                # ある別の原子とのつながりを見る
                for j in range(i):
                    if n_bond_mtx[ind, i, j] == 0:
                        continue
                    # C同士の４重結合
                    if n_bond_mtx[ind, i, j] > 3:
                        break
                    # C同士が繋がっていて、ラジカルの差が指定以上なら、除外
                    if abs(n_bond_mtx[ind, j, j] - n_bond_mtx[ind, i, i])>self.max_bias_rad_num:
                        break
                    else:
                        # 正常にfor、もしくはcontinueされればここに飛ぶ
                        continue
                    break
                else:
                    # 正常にfor、もしくはcontinueされればここに飛ぶ
                    continue
                # C同士が繋がっていて、ラジカルの差が2以上ならここに飛ぶ
                # もしくはCしかない化合物
                break
            else:
                # 正常にfor、もしくはcontinueされればここに飛ぶ
                # OKな原子はno_bias_listに追加。
                no_bias_bool[ind]=True
                continue

        return no_bias_bool

    def bias_check_o3(
            self,
            n_bond_mtx,
            cand_num,
            vlc_mt_4):
        """
        原子価4の原子の数が2以下なら
        C同士が繋がっていて、ラジカルの差が指定以上か、
        原子価4同士の結合の差が指定以上か、
        Cしかない化合物の場合を除外
        """
        vlc4_id_not=np.logical_not(np.identity(vlc_mt_4,bool))
        vlc4_sel=np.zeros([vlc_mt_4,self.atm_num],bool)
        vlc4_sel[:,:vlc_mt_4]=vlc4_id_not
        no_bias_bool=[False]*cand_num
        for ind in range(cand_num):
            for i in range(vlc_mt_4):
                # 原子価4以上、自分以外、結合数0以上の注目bond
                atn_bond=np.logical_and(n_bond_mtx[ind,i,:]>0,vlc4_sel[i,:])
                vlc4_minor_bond=min(n_bond_mtx[ind,i,atn_bond])
                vlc4_major_bond=max(n_bond_mtx[ind,i,atn_bond])
                # 原子価4同士の結合の差が指定以上なら
                if (vlc4_major_bond-vlc4_minor_bond)>self.max_bias_bonds:
                    break
                # ある別の原子とのつながりを見る
                atn_bonds=np.zeros(self.atm_num,bool)
                for j in range(i):
                    if n_bond_mtx[ind, i, j] == 0:
                        continue
                    # C同士の４重結合
                    if n_bond_mtx[ind, i, j] > 3:
                        break
                    # C同士が繋がっていて、ラジカルの差が指定以上なら、除外
                    if abs(n_bond_mtx[ind, j, j] - n_bond_mtx[ind, i, i])>self.max_bias_rad_num:
                        break
                    else:
                        continue
                    break
                else:
                    # 正常にfor、もしくはcontinueされればここに飛ぶ
                    continue
                # C同士が繋がっていて、ラジカルの差が2以上か、
                # 原子価4同士の結合の差が指定以上か、
                # Cしかない化合物の場合ここに飛ぶ
                break
            else:
                # 正常にfor、もしくはcontinueされればここに飛ぶ
                # OKな原子はno_bias_listに追加。
                no_bias_bool[ind]=True
                continue

        return no_bias_bool


    def iter_display_graph(self, limit_no=10):
        """
        networkxの機能でグラフを描写する
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        save_to_graph = GraphSet.converters["to_graph"]
        GraphSet.converters["to_graph"] = nx.Graph
        try:
            for path in self.paths:

                node_color = []
                for node in path.nodes:
                    node_color.append(
                        self.atm_colors_list[node])
                nx.draw_networkx(
                    path,
                    node_color=node_color)
                plt.show()

                limit_no -= 1
                if limit_no == 0:
                    break
        finally:
            GraphSet.converters["to_graph"] = save_to_graph
