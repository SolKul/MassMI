import numpy as np
import matplotlib.pyplot as plt

class BayesianOptDiscrete:
    def __init__(self,func,features,lam=1.5):
        self.t_1=1
        self.t_2=0.4
        self.t_3=0.1
        self.lam=lam
        
        self.func=func
        self.features=features# 特徴量ベクトル:x
        self.target_num=len(features)

        # すべてのtargetの連番のリスト
        self.target_no_list=list(range(0,self.target_num))
        # まだyを計算していないtargetのnoのリスト
        self.remain_target_no_list=self.target_no_list.copy()
        
    def minimumize_func(self,niter=10):
        self.initial_estimate()
        for i in range(niter):
            self.calculate_kernel()
            self.calculate_e_v()
            self.calculate_lcb()
            self.append_next_x()
          
        min_ind=np.argmin(self.train_y)
        self.min_target_no=self.train_target_no_list[min_ind]
        self.min_y=self.train_y[min_ind]
        return self.min_y

    def initial_estimate(self,initial_num=2):
        """
        初期点でのyの計算
        """
        # 初期点をランダムにチョイス
        train_target_no=np.random.choice(
            np.arange(0,self.target_num),
            initial_num,
            replace=False)
        self.train_target_no_list=train_target_no.tolist()

        # 選んだ点をyを計算していないtargetのnoのリストから削除
        for i in range(initial_num):
            self.remain_target_no_list.remove(self.train_target_no_list[i])

        # 特徴量ベクトルを格納
        self.train_x=self.features[self.train_target_no_list,:]

        # yの値を格納
        self.train_y_list=[]
        for i in range(initial_num):
            y=self.func(self.train_target_no_list[i])
            self.train_y_list.append(y)
        self.train_y=np.array(self.train_y_list)

        # 標準化
        self.mean_train_y=np.mean(self.train_y)
        self.std_train_y=np.std(self.train_y,ddof=1)
        self.stdrized_train_y=(self.train_y-self.mean_train_y)/self.std_train_y
        
    def calculate_kernel(self):

        """
        カーネルの計算
        """
        # 選んだ点についてカーネルを計算
        num_train=len(self.train_x)

        K_dis=np.sum((self.train_x[np.newaxis,:,:]-self.train_x[:,np.newaxis,:])**2,axis=2)
        K=self.t_1*np.exp(-K_dis/self.t_2) \
            +np.eye(num_train)*self.t_3
        self.K_i=np.linalg.pinv(K)
        self.K_i_y=self.K_i@self.stdrized_train_y
        
    def calculate_e_v(self):

        """
        期待値と分散の計算
        """
        # yを計算していないtargetのnoのリストの特徴量ベクトルを取得
        remain_x=features[self.remain_target_no_list,:]
        remain_num=len(remain_x)
        self.expected_y=np.zeros(remain_num)
        self.variance_y=np.zeros(remain_num)
        # 一つずつ、期待値と分散を計算していく
        for i in range(remain_num):
            k_st_dis=np.sum((self.train_x-remain_x[i:i+1,:])**2,axis=1)
            k_st=self.t_1*np.exp(-k_st_dis/self.t_2)
            s=self.t_1+self.t_3
            self.expected_y[i]=k_st@self.K_i_y
            self.variance_y[i]=s-k_st@self.K_i@k_st
            
    def calculate_lcb(self):
    
        """
        LCB(lower confidence bound)を計算
        """
        std_y=np.sqrt(self.variance_y)
        num_train=len(self.train_x)
        self.alpha_lcb=-self.expected_y+self.lam*(np.sqrt(np.log(num_train)/num_train))*std_y
        
    def append_next_x(self):
        """
        次のxを決定する。
        """
        target_no=np.argmax(self.alpha_lcb)
        next_target_no=self.remain_target_no_list[target_no]
        self.train_target_no_list.append(next_target_no)
        
        # 選んだ点をyを計算していないtargetのnoのリストから削除
        self.remain_target_no_list.remove(next_target_no)
        
        # 特徴量ベクトルを格納
        self.train_x=self.features[self.train_target_no_list,:]
        
        self.train_y_list.append(self.func(next_target_no))
        self.train_y=np.array(self.train_y_list)

        # 標準化
        self.mean_train_y=np.mean(self.train_y)
        self.std_train_y=np.std(self.train_y,ddof=1)
        self.stdrized_train_y=(self.train_y-self.mean_train_y)/self.std_train_y
        
    def plot_result(self):
        # 訓練データ数
        train_num=len(self.train_target_no_list)
        train_no_arange=np.arange(train_num)

        # 未評価データ数
        remain_num=len(self.remain_target_no_list)
        remain_no_arange=np.arange(train_num,train_num+remain_num)

        # 標準化された期待値に標準偏差を掛け、平均を足して訓練データと合わせる
        scaled_expected_y=self.expected_y[:-1]*self.std_train_y \
            +self.mean_train_y
        # 標準化されたvarianceに標準偏差の2乗を掛け、訓練データと合わせる
        scaled_variance_y=self.variance_y[:-1]*self.std_train_y**2
        # 獲得関数
        alpha=self.alpha_lcb[:-1]
        # 獲得関数の大きい順(降順)に並び替える
        descending=np.argsort(alpha)[::-1]

        plt.figure(figsize=(10,5))
        plt.plot(train_no_arange,self.train_y,label="True Value")
        plt.plot(remain_no_arange,scaled_expected_y[descending],label="Expect")
        plt.plot(remain_no_arange,scaled_variance_y[descending],label="Variance")
        plt.plot(remain_no_arange,alpha[descending],label="$\\alpha_{lcb}$")
        plt.ylabel("y")
        plt.legend()
        plt.show()
        
    def plot_trasition(self):
        train_num=len(self.train_y)
        min_y_trans=np.zeros(train_num)
        for i in range(train_num):
            min_y_trans[i]=np.min(self.train_y[:i+1])
        plt.plot(min_y_trans)
        plt.show()