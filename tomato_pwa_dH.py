import numpy as np
import statsmodels.api as sm
from scipy import linalg, optimize
from sklearn.svm import SVC

H_0 = 62.9
H_plusinf = 43.1
H_minusinf = 124
k_ref = 0.0019
E_a = 170.604
T_ref = 288.15
Rg = 0.008314

T_min, T_max = 278, 298
H_min, H_max = 42, H_0
Enz_min, Enz_max = 61, 80

def k_rate(T):
    return k_ref*np.exp((E_a/Rg)*(1/T_ref-1/T))

def f(T,H,Enz):
    return -k_rate(T)*H*Enz


n = 3
N = 600
c = 5
s = 5
num_piece = int(s*(s-1)/2)


# row vector(1-dim array)
T = np.random.uniform(T_min,T_max,(N,))
H = np.random.uniform(H_min,H_max,(N,))
Enz = np.random.uniform(Enz_min,Enz_max,(N,))

# X : n×N dimension matrix, y : N dimension row vector(1-dim array)
X = np.array([T,H,Enz])
y = f(T,H,Enz)

# Norm  :N×N dimension matrix, Norm_sort : N×N dimension matrix, C : N×c dimension matrix
Norm = np.empty((X.shape[1], X.shape[1]))
for i in range(X.shape[1]):
    for j in range(X.shape[1]):
        Norm[i, j] = np.linalg.norm(X[:, i] - X[:,j])
Norm_sort = np.argsort(Norm, axis=1)
C = Norm_sort[:, 0:c]

# Xc : N×n×c dimension matrix, T_c,H_c,Enz_c : N×c dimension matrix, Yc : N×c×1 dimension matrix
Xc = np.empty((C.shape[0], X.shape[0], C.shape[1]))
for i in range(Xc.shape[0]):
    for j in range(Xc.shape[2]):
        Xc[i, :, j] = X[:, C[i, j]]
T_c = Xc[:, 0, :]
H_c = Xc[:, 1, :]
Enz_c = Xc[:, 2, :]

yc = f(T_c,H_c,Enz_c)
Yc = yc[:, :, np.newaxis]

# Phi : N×c×(n+1) dimension matrix, (Phi)'×Phi : N×2×2 dimension matrix, inverse matrix of phi : N×c×c dimension matrix, theta_ls : N×(n+1)×1 dimension matrix
one = np.ones((Xc.shape[0], 1, Xc.shape[2]))
Phi_T = np.concatenate((Xc, one), axis=1)
Phi = Phi_T.transpose(0, 2, 1)
phi = Phi_T @ Phi
inv_phi = np.linalg.inv(phi)
PHI = inv_phi @ Phi_T
Theta_ls = PHI @ Yc

# SSR : N dimension vector, m : N×n dimension matrix, V : N×(n+1)×(n+1) dimension matrix, Q : N×n×n dimension matrix, eps : feature vector(N×(2n+1) dimension matrix), R : N×(2n+1)×(2n+1) matrix, w : N dimension vector (1dim-array)
eye = np.stack(([np.eye(Xc.shape[2])]*Xc.shape[0]), axis = 0)
SSR = Yc.transpose(0, 2, 1) @ (eye - (Phi @ PHI)) @ Yc
m = np.sum(Xc, axis=2)/c
V = (SSR/(c-n-1)) * inv_phi
Q = (Xc-m[:,:,np.newaxis]) @ (Xc-m[:,:,np.newaxis]).transpose(0,2,1)
theta_ls_H = Theta_ls.transpose(0, 2, 1)
eps = np.empty((Xc.shape[0], 2*Xc.shape[1]+1))
for i in range(Xc.shape[0]):
    eps[i, :] = np.concatenate((theta_ls_H[i, :, :].flatten(),m[i, :]), axis = 0)
Zero_upper = np.zeros((V.shape[0], V.shape[1], Q.shape[2]))
Zero_lower = np.zeros((Q.shape[0], Q.shape[1], V.shape[2]))
Upper = np.concatenate((V, Zero_upper), axis=2)
Lower = np.concatenate((Zero_lower, Q), axis=2)
R = np.concatenate((Upper,Lower), axis=1)
pai = np.power(2*np.pi, 2*n+1)
det_R = np.linalg.det(R)
re_w = np.sqrt(pai*det_R)
w = 1/re_w



class FMeans_pp:
    def __init__(self, n_clusters, max_iter = 1000, random_seed = 0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)

    def fit(self, X, R):
        #ランダムに最初のクラスタ点を決定
        # X : N×(2n+1) matrix, R : N×(2n+1)×(2n+1) matrix(array), tmp : scalar
        tmp = np.random.choice(np.array(range(X.shape[0])))
        first_cluster = X[tmp]
        first_cluster = first_cluster[np.newaxis,:]
        determinant = []
        for i in range(N):
            determinant.append(int(np.linalg.det(R[i,:,:])))
        # print(determinant)
        Rinv = np.linalg.inv(R)
        

        #最初のクラスタ点とそれ以外のデータ点との行列ノルムの2乗を計算し、それぞれをその総和で割る
        # ∥ X - first_cluster ∥_Rinv^2 = <X-first_cluster, Rinv(x-first_cluster)>
        #X-first_cluster : N×(2n+1) matrix, Rinv(x-first_cluster) : N×(2n+1)×1 matrix(array)
        left_vec = X - first_cluster
        right_vec = Rinv @ left_vec[:,:,np.newaxis]
        #norm_m = left_vec[:,np.newaxis,:] @ right_vec
        #dist_p = np.diagonal(norm_m, axis1 = 1, axis2 = 2)
        # p : N dimension vector(1-dim array)
        p = ((left_vec[:,np.newaxis,:] @ right_vec) / (left_vec[:,np.newaxis,:] @ right_vec).sum()).reshape(X.shape[0],)

        r =  np.random.choice(np.array(range(X.shape[0])), size = 1, replace = False, p = p)
        first_cluster = np.r_[first_cluster ,X[r]]

        #分割するクラスター数が3個以上の場合
        if self.n_clusters >= 3:
            #指定の数のクラスタ点を指定できるまで繰り返し
            while first_cluster.shape[0] < self.n_clusters:
                #各クラスター点と各データポイントとの行列ノルムの2乗を算出
                #dist_f : N×s matrix
                left_v = (X[:, :, np.newaxis] - first_cluster.T[np.newaxis, :, :]).transpose(0, 2, 1)
                right_v = Rinv @ (X[:, :, np.newaxis] - first_cluster.T[np.newaxis, :, :])
                norm_m = left_v @ right_v
                dist_f = np.diagonal(norm_m, axis1 = 1, axis2 =2)
                #print('dist_f(pre):', dist_f)
                dist_f.flags.writeable = True
                #最も距離の近いクラスター点はどれか導出
                f_argmin = dist_f.argmin(axis = 1)
                #最も距離の近いクラスター点と各データポイントとの行列ノルムの2乗を導出
                #属しないクラスター点との距離を0にする
                for i in range(dist_f.shape[1]):
                    dist_f.T[i][f_argmin != i] = 0
                #print('dist_f:', dist_f)

                #新しいクラスタ点を確率的に導出
                pp = dist_f.sum(axis = 1) / dist_f.sum()

                rr = np.random.choice(np.array(range(X.shape[0])), size = 1, replace = False, p = pp)
                #新しいクラスター点を初期値として加える
                first_cluster = np.r_[first_cluster ,X[rr]]
        # print('first_cluster:', first_cluster)

        #最初のラベルづけを行う
        left = (X[:, :, np.newaxis] - first_cluster.T[np.newaxis, :, :]).transpose(0, 2, 1)
        right = Rinv @ (X[:, :, np.newaxis] - first_cluster.T[np.newaxis, :, :])
        norm = left @ right
        dist = np.diagonal(norm, axis1 = 1, axis2 =2)
        #dist = (((X[:, :, np.newaxis] - first_cluster.T[np.newaxis, :, :]) ** 2).sum(axis = 1))
        
        self.labels_ = dist.argmin(axis = 1)
        # print('labels(first):', self.labels_)
        labels_prev = np.zeros(X.shape[0])
        count = 0
        self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))

        #各データポイントが属しているクラスターが変化しなくなった、又は一定回数の繰り返しを越した場合は終了
        while (not (self.labels_ == labels_prev).all() and count < self.max_iter):
            #update the centers of the cluster
            for i in range(self.n_clusters):
                XX = X[self.labels_ == i, :]
                RR = Rinv[self.labels_ == i, :, :]
                # print('RR:', RR.shape)
                RRinv = np.linalg.inv(RR.sum(axis=0))
                self.cluster_centers_[i, :] = (RRinv @ ((RR @ XX[:,:,np.newaxis]).sum(axis=0))).T

            # print('cluster_centers:', self.cluster_centers_)
            
            #各データポイントと各クラスター中心間の行列ノルムを総当たりで計算する
            # dist : N×s dimension matrix
            Left_v = (X[:,:,np.newaxis] - self.cluster_centers_.T[np.newaxis,:,:]).transpose(0,2,1)
            Right_v = Rinv @ (X[:,:,np.newaxis] - self.cluster_centers_.T[np.newaxis,:,:])
            Norm = Left_v @ Right_v
            dist = np.diagonal(Norm, axis1 = 1, axis2 =2)
            
            #1つ前のクラスターラベルを覚えておく。1つ前のラベルとラベルが変化しなければプログラムは終了する。
            labels_prev = self.labels_
            #再計算した結果、最も距離の近いクラスターのラベルを割り振る
            self.labels_ = dist.argmin(axis = 1)
            # print('labels:', self.labels_)
            count += 1
            self.count = count
            # print('count:', self.count)

    def predict(self, X):
        dist = ((X[:, :, np.newaxis] - self.cluster_centers_.T[np.newaxis, :, :]) ** 2).sum(axis = 1)
        labels = dist.argmin(axis = 1)
        return labels


print('----------------------------------------------------------------------------------------')
print('-----------------------------------f_H(t)の線形回帰--------------------------------------')
print('----------------------------------------------------------------------------------------')

model = FMeans_pp(s)
model.fit(eps, R)

Xr0 = Xc[model.labels_ == 0,:,:]
wr0 = w[model.labels_ == 0]
#print('wr0:', wr0.shape)
X0 = Xr0[0,:,:]
w0 = [wr0[0]]*c
for i in range(Xr0.shape[0]-1):
    i += 1
    X0 = np.c_[X0, Xr0[i,:,:]]
    w0 = np.r_[w0, [wr0[i]]*c]
#print('X0:', X0.shape)
#print('w0:', w0.shape)
T0 = X0[0,:]
H0 = X0[1,:]
Enz0 = X0[2,:]
y0 = f(T0, H0, Enz0)

Xr1 = Xc[model.labels_ == 1,:,:]
wr1 = w[model.labels_ == 1]
#print('wr1:', wr1.shape)
X1 = Xr1[0,:,:]
w1 = [wr1[0]]*c
for i in range(Xr1.shape[0]-1):
    i += 1
    X1 = np.c_[X1, Xr1[i,:,:]]
    w1 = np.r_[w1, [wr1[i]]*c]
#print('X1:', X1.shape)
#print('w1:', w1.shape)
T1 = X1[0,:]
H1 = X1[1,:]
Enz1 = X1[2,:]
y1 = f(T1, H1, Enz1)

Xr2 = Xc[model.labels_ == 2,:,:]
wr2 = w[model.labels_ == 2]
#print('wr2:', wr2.shape)
X2 = Xr2[0,:,:]
w2 = [wr2[0]]*c
for i in range(Xr2.shape[0]-1):
    i += 1
    X2 = np.c_[X2, Xr2[i,:,:]]
    w2 = np.r_[w2, [wr2[i]]*c]
#print('X2:', X2.shape)
#print('w2:', w2.shape)
T2 = X2[0,:]
H2 = X2[1,:]
Enz2 = X2[2,:]
y2 = f(T2, H2, Enz2)

Xr3 = Xc[model.labels_ == 3,:,:]
wr3 = w[model.labels_ == 3]
#print('wr3:', wr3.shape)
X3 = Xr3[0,:,:]
w3 = [wr3[0]]*c
for i in range(Xr3.shape[0]-1):
    i += 1
    X3 = np.c_[X3, Xr3[i,:,:]]
    w3 = np.r_[w3, [wr3[i]]*c]
#print('X3:', X3.shape)
#print('w3:', w3.shape)
T3 = X3[0,:]
H3 = X3[1,:]
Enz3 = X3[2,:]
y3 = f(T3, H3, Enz3)

Xr4 = Xc[model.labels_ == 4,:,:]
wr4 = w[model.labels_ == 4]
#print('wr4:', wr4.shape)
X4 = Xr4[0,:,:]
w4 = [wr4[0]]*c
for i in range(Xr4.shape[0]-1):
    i += 1
    X4 = np.c_[X4, Xr4[i,:,:]]
    w4 = np.r_[w4, [wr4[i]]*c]
#print('X4:', X4.shape)
#print('w4:', w4.shape)
T4 = X4[0,:]
H4 = X4[1,:]
Enz4 = X4[2,:]
y4 = f(T4, H4, Enz4)


F = [X0, X1, X2, X3, X4]
L = [y0, y1, y2, y3, y4]
W = [w0, w1, w2, w3, w4]

theta = np.empty((s, n+1))
A_c, B_c, C_c, D_c = np.empty(s), np.empty(s), np.empty(s), np.empty(s)

for i in range(s):
    Fe = sm.add_constant(F[i].T)
    mod_wls = sm.WLS(L[i], Fe, weights=W[i])
    res_wls = mod_wls.fit()
    theta[i,:] = res_wls.params

for i in range(s):
    D_c[i] = theta[i,0]
    C_c[i] = theta[i,1]
    A_c[i] = theta[i,2]
    B_c[i] = theta[i,3]

print('------------------------------------------------------------------------------------------------------------------')
# print('[D:y切片, C:T(t)の係数, A:H(t)の係数, B:Enz(t)の係数]:\n', theta)
print('A:H(t)の係数:\n', A_c)
print('B:Enz(t)の係数:\n', B_c)
print('C:T(t)の係数:\n', C_c)
print('D:y切片:\n', D_c)


print('----------------------------------------------------------------------------------------')
print('-----------------------------------f_H(t)の区分領域--------------------------------------')
print('----------------------------------------------------------------------------------------')

X_features = np.concatenate([X0, X1, X2, X3, X4], 1)
X_labels = np.concatenate((np.array([0]*X0.shape[1]), np.array([1]*X1.shape[1]), np.array([2]*X2.shape[1]), np.array([3]*X3.shape[1]),np.array([4]*X4.shape[1])), axis = 0)
clf = SVC(kernel='linear', decision_function_shape='ovo')
clf.fit(X_features.T, X_labels)
Norm_SV_ID = clf.decision_function(clf.support_vectors_)
Num_SV = clf.n_support_

Q_p, T_p, R_p, S_p = np.empty(num_piece), np.empty(num_piece), np.empty(num_piece), np.empty(num_piece)

for i in range(num_piece):
    R_p[i] = clf.coef_[i,0]
    Q_p[i] = clf.coef_[i,1]
    T_p[i] = clf.coef_[i,2]
S_p = clf.intercept_

for i in range(s):
    print('Number of label %d : %d' % (i, F[i].shape[1]))

# print('X_features:', X_features.shape)
# print('X_labels:', X_labels.shape)
# print('coef_ID function(R:T(t)の係数, Q:H(t)の係数, T:Enz(t)の係数):\n', clf.coef_)
print('Q:H(t)の係数 :\n', Q_p)
print('T:Enz(t)の係数) :\n', T_p)
print('R:T(t)の係数 :\n', R_p)
print('intercept_ID function(S):\n', clf.intercept_)
# print('support_index:\n', clf.support_)
# print('Number_SupportVector:\n', Num_SV)
# print('SupportVectors:\n', clf.support_vectors_)
# print('Norm between SupportVector and ID_function:\n', Norm_SV_ID)
# print('Norm in class0:\n', Norm_SV_ID[Num_SV[0]-10:Num_SV[0]-1])
# print('Norm in class1:\n', Norm_SV_ID[Num_SV[0]-1+Num_SV[1]-10:Num_SV[0]-1+Num_SV[1]-1])
# print('Norm in class2:\n', Norm_SV_ID[Num_SV[0]+Num_SV[1]-1+Num_SV[2]-10:Num_SV[0]+Num_SV[1]-1+Num_SV[2]-1])



#===================================================#
#======== Caluculate h =============================#
#===================================================#


def pwa(H, Enz, Ta, A, B, C, D):
    return A*H + B*Enz + C*Ta + D

def get_gmin(s,A,B,C,D):
    gmin = np.empty(s)
    for i in range(s):
        choice = [pwa(H_min,Enz_min,T_min,A[i],B[i],C[i],D[i]), pwa(H_min,Enz_min,T_max,A[i],B[i],C[i],D[i]),
        pwa(H_min,Enz_max,T_min,A[i],B[i],C[i],D[i]), pwa(H_min,Enz_max,T_max,A[i],B[i],C[i],D[i]),
        pwa(H_max,Enz_min,T_min,A[i],B[i],C[i],D[i]), pwa(H_max,Enz_min,T_max,A[i],B[i],C[i],D[i]),
        pwa(H_max,Enz_max,T_min,A[i],B[i],C[i],D[i]), pwa(H_max,Enz_max,T_max,A[i],B[i],C[i],D[i])]
        gmin[i] = min(choice)
    return gmin


def get_gmax(s,A,B,C,D):
    gmax = np.empty(s)
    for i in range(s):
        choice = [pwa(H_min,Enz_min,T_min,A[i],B[i],C[i],D[i]), pwa(H_min,Enz_min,T_max,A[i],B[i],C[i],D[i]),
        pwa(H_min,Enz_max,T_min,A[i],B[i],C[i],D[i]), pwa(H_min,Enz_max,T_max,A[i],B[i],C[i],D[i]),
        pwa(H_max,Enz_min,T_min,A[i],B[i],C[i],D[i]), pwa(H_max,Enz_min,T_max,A[i],B[i],C[i],D[i]),
        pwa(H_max,Enz_max,T_min,A[i],B[i],C[i],D[i]), pwa(H_max,Enz_max,T_max,A[i],B[i],C[i],D[i])]
        gmax[i] = max(choice)
        # print(choice)
    return gmax


print("gmin:\n", get_gmin(s,A_c,B_c,C_c,D_c))
print("gmax:\n", get_gmax(s,A_c,B_c,C_c,D_c))
print("hmin:\n", get_gmin(num_piece,Q_p,T_p,R_p,S_p))
print("hmax:\n", get_gmax(num_piece,Q_p,T_p,R_p,S_p))




#==============================================================================================#
#============================区分アフィン関数=====================================================#
#==============================================================================================#

def fun0(Ta,H,Enz,A,B,C,D):
    return A[0]*H + B[0]*Enz + C[0]*Ta + D[0]
def fun1(Ta,H,Enz,A,B,C,D):
    return A[1]*H + B[1]*Enz + C[1]*Ta + D[1]
def fun2(Ta,H,Enz,A,B,C,D):
    return A[2]*H + B[2]*Enz + C[2]*Ta + D[2]
def fun3(Ta,H,Enz,A,B,C,D):
    return A[3]*H + B[3]*Enz + C[3]*Ta + D[3]
def fun4(Ta,H,Enz,A,B,C,D):
    return A[4]*H + B[4]*Enz + C[4]*Ta + D[4]

def state_partion(Ta,H,Enz,Q,T,R,S):
    if Q[0]*H+T[0]*Enz+R[0]*Ta+S[0] >=0 and Q[1]*H+T[1]*Enz+R[1]*Ta+S[1] >=0 and Q[2]*H+T[2]*Enz+R[2]*Ta+S[2] >=0 and Q[3]*H+T[3]*Enz+R[3]*Ta+S[3] >=0:
        return 0
    elif Q[0]*H+T[0]*Enz+R[0]*Ta+S[0] <=0 and Q[4]*H+T[4]*Enz+R[4]*Ta+S[4] >=0 and Q[5]*H+T[5]*Enz+R[5]*Ta+S[5] >=0 and Q[6]*H+T[6]*Enz+R[6]*Ta+S[6] >=0:
        return 1
    elif Q[1]*H+T[1]*Enz+R[1]*Ta+S[1] <=0 and Q[4]*H+T[4]*Enz+R[4]*Ta+S[4] <=0 and Q[7]*H+T[7]*Enz+R[7]*Ta+S[7] >=0 and Q[8]*H+T[8]*Enz+R[8]*Ta+S[8] >=0:
        return 2
    elif Q[2]*H+T[2]*Enz+R[2]*Ta+S[2] <=0 and Q[5]*H+T[5]*Enz+R[5]*Ta+S[5] <=0 and Q[7]*H+T[7]*Enz+R[7]*Ta+S[7] <=0 and Q[9]*H+T[9]*Enz+R[9]*Ta+S[9] >=0:
        return 3
    elif Q[3]*H+T[3]*Enz+R[3]*Ta+S[3] <=0 and Q[6]*H+T[6]*Enz+R[6]*Ta+S[6] <=0 and Q[8]*H+T[8]*Enz+R[8]*Ta+S[8] <=0 and Q[9]*H+T[9]*Enz+R[9]*Ta+S[9] <=0:
        return 4
    else:
        return 5

#==============================================================================================#
#==========================誤差を計算して表示=====================================================#
#==============================================================================================#

def error(Ta,H,Enz,A,B,C,D,Q,T,R,S):
    if state_partion(Ta,H,Enz,Q,T,R,S) == 0:
        e = (f(Ta,H,Enz)-fun0(Ta,H,Enz,A,B,C,D))**2
        er = np.sqrt(e)
        print(er)
    elif state_partion(Ta,H,Enz,Q,T,R,S) == 1:
        e = (f(Ta,H,Enz)-fun1(Ta,H,Enz,A,B,C,D))**2
        er = np.sqrt(e)
        print(er)
    elif state_partion(Ta,H,Enz,Q,T,R,S) == 2:
        e = (f(Ta,H,Enz)-fun2(Ta,H,Enz,A,B,C,D))**2
        er = np.sqrt(e)
        print(er)
    elif state_partion(Ta,H,Enz,Q,T,R,S) == 3:
        e = (f(Ta,H,Enz)-fun3(Ta,H,Enz,A,B,C,D))**2
        er = np.sqrt(e)
        print(er)
    elif state_partion(Ta,H,Enz,Q,T,R,S) == 4:
        e = (f(Ta,H,Enz)-fun4(Ta,H,Enz,A,B,C,D))**2
        er = np.sqrt(e)
        print(er)
    else:
        print("error")


N_check = 5
Ta_check = np.random.uniform(T_min,T_max,(N_check,))
H_check = np.random.uniform(H_min,H_max,(N_check,))
Enz_check = np.random.uniform(Enz_min,Enz_max,(N_check,))

print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("error : non-linear - linear")
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
for i in range(H_check.shape[0]):
    for j in range(Enz_check.shape[0]):
        for k in range(Ta_check.shape[0]):
            error(Ta_check[k],H_check[i],Enz_check[j],A_c,B_c,C_c,D_c,Q_p,T_p,R_p,S_p)