import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns
import tensorflow as tf
import numpy as np
import timeit
import xlrd
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

# PINN model
class PINN_model:
    def __init__(self, layers, X, X_b):
        
        self.max_X, self.min_X = X.max(0), X.min(0)


        self.x, self.y = X[:, 0:1], X[:, 1:2]
        self.x_b_1, self.y_b_1 = X_b[0][:, 0:1], X_b[0][:, 1:2]
        self.x_b_2, self.y_b_2 = X_b[1][:, 0:1], X_b[1][:, 1:2]
        self.x_b_3, self.y_b_3 = X_b[2][:, 0:1], X_b[2][:, 1:2]
        self.x_b_4, self.y_b_4 = X_b[3][:, 0:1], X_b[3][:, 1:2]
        self.q = -10.0


        
        self.v = 0.0
        self.E = 12000000.0
        self.t = 0.01
        self.D0 = (self.E*self.t**3) / (12*(1-self.v**2))
        self.D = self.D0 * tf.constant([[1.0, self.v, 0.0],
                                        [self.v, 1.0, 0.0],
                                        [0.0, 0.0, (1-self.v)/2.0]])
        
        
        
        self.layers_u, self.layers_k, self.layers_M = layers[0], layers[1], layers[2]
        self.weights_u, self.biases_u = self.initialize_NN(self.layers_u)
        self.weights_k, self.biases_k = self.initialize_NN(self.layers_k)
        self.weights_M, self.biases_M = self.initialize_NN(self.layers_M)
        
        
        
         
        self.x_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.y_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.x_b_1_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.y_b_1_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.x_b_2_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.y_b_2_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.x_b_3_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.y_b_3_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.x_b_4_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.y_b_4_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])


        
        self.u_pred = self.net_u(self.x_tf, self.y_tf)
        self.k_pred = self.net_k(self.x_tf, self.y_tf)
        self.M_pred = self.net_M(self.x_tf, self.y_tf)
        
        
        self.u_b_1_pred = self.net_u(self.x_b_1_tf, self.y_b_1_tf)
        self.u_b_2_pred = self.net_u(self.x_b_2_tf, self.y_b_2_tf)
        self.u_b_3_pred = self.net_u(self.x_b_3_tf, self.y_b_3_tf)
        self.u_b_4_pred = self.net_u(self.x_b_4_tf, self.y_b_4_tf)

        
        self.M_b_1_pred = self.net_M(self.x_b_1_tf, self.y_b_1_tf)
        self.M_b_2_pred = self.net_M(self.x_b_2_tf, self.y_b_2_tf)
        self.M_b_3_pred = self.net_M(self.x_b_3_tf, self.y_b_3_tf)
        self.M_b_4_pred = self.net_M(self.x_b_4_tf, self.y_b_4_tf)


        self.f_u_k_pred, self.f_k_M_pred, self.f_M_q_pred = self.net_f(self.x_tf, self.y_tf)

        
        
        self.loss_b_u = tf.reduce_mean(tf.square(self.u_b_1_pred)) \
            + tf.reduce_mean(tf.square(self.u_b_2_pred)) \
            + tf.reduce_mean(tf.square(self.u_b_3_pred)) \
            + tf.reduce_mean(tf.square(self.u_b_4_pred))
           

        
        '''  
        self.loss_bc_a = tf.reduce_mean(tf.square(self.a_b_1_pred)) \
            + tf.reduce_mean(tf.square(self.a_b_2_pred)) \
            + tf.reduce_mean(tf.square(self.a_b_3_pred)) \
            + tf.reduce_mean(tf.square(self.a_b_4_pred))

        self.loss_bc_M = tf,reduce_mean(tf.square(self.M_b_1_pred)) \
            + tf.reduce_mean(tf.square(self.M_b_2_pred)) \
            + tf.reduce_mean(tf.square(self.M_b_3_pred)) \
            + tf.reduce_mean(tf.square(self.M_b_4_pred)) 
        '''
        
        
        
        self.loss_c = 1e3*self.loss_b_u \
            + tf.reduce_mean(tf.square(self.M_b_1_pred[:, 1:2])) \
            + tf.reduce_mean(tf.square(self.M_b_2_pred[:, 1:2])) \
            + tf.reduce_mean(tf.square(self.M_b_3_pred[:, 0:1])) \
            + tf.reduce_mean(tf.square(self.M_b_4_pred[:, 0:1])) \
            + tf.reduce_mean(tf.square(self.M_b_1_pred[:, 2:3])) \
            + tf.reduce_mean(tf.square(self.M_b_2_pred[:, 2:3])) \
            + tf.reduce_mean(tf.square(self.M_b_3_pred[:, 2:3])) \
            + tf.reduce_mean(tf.square(self.M_b_4_pred[:, 2:3]))          
        
        '''
        self.loss_c = 1000*self.loss_b_u \
            + tf.reduce_mean(tf.square(self.a_b_1_pred[:, 1:2])) \
            + tf.reduce_mean(tf.square(self.a_b_2_pred[:, 1:2])) \
            + tf.reduce_mean(tf.square(self.a_b_3_pred[:, 0:1])) \
            + tf.reduce_mean(tf.square(self.a_b_4_pred[:, 0:1]))
        '''


        self.loss_f = tf.reduce_mean(tf.square(self.f_u_k_pred)) \
            + tf.reduce_mean(tf.square(self.f_k_M_pred)) \
            + tf.reduce_mean(tf.square(self.f_M_q_pred))


        
        self.loss = self.loss_c +  self.loss_f
          
        
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)


        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})


        
        self.loss_c_log = []
        self.loss_f_log = []
        self.saver = tf.train.Saver()
        
        
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    


    
    
    def xavier_init(self,size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

    
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

   
    def forward_pass(self, H, weights, biases, layers):

        num_layers = len(layers)
                   
        #H = (H - self.mean_X)/self.std_X
        H = 2*(H - self.min_X)/(self.max_X - self.min_X) - 1
        
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H



    
    def net_u(self, x, y):  
        u = self.forward_pass(tf.concat([x, y], 1), self.weights_u, self.biases_u, self.layers_u)
        return u


    def net_k(self, x, y):  
        k = self.forward_pass(tf.concat([x, y], 1), self.weights_k, self.biases_k, self.layers_k)
        return k
    
    def net_M(self, x, y):  
        M = self.forward_pass(tf.concat([x, y], 1), self.weights_M, self.biases_M, self.layers_M)
        return M



    def net_f(self, x, y):
        u = self.net_u(x, y)
        k = self.net_k(x, y)
        M = self.net_M(x, y)
        
        # u - k
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_xy = tf.gradients(u_x, y)[0]
        f_u_k = tf.concat([-u_xx, -u_yy, -2*u_xy], 1) - k
        
        # k - M
        f_k_M = tf.matmul(k, self.D) - M
        
        # M - q
        Mx, My, Mxy = M[:, 0:1], M[:, 1:2], M[:, 2:3]
        
        Mx_x = tf.gradients(Mx, x)[0]
        Mx_xx = tf.gradients(Mx_x, x)[0]
        My_y = tf.gradients(My, y)[0]
        My_yy = tf.gradients(My_y, y)[0]
        Mxy_x = tf.gradients(Mxy, x)[0]
        Mxy_xy = tf.gradients(Mxy_x, y)[0]
        
        f_M_q = Mx_xx + My_yy + 2*Mxy_xy + self.q
        
        return f_u_k, f_k_M, f_M_q







   
    def train(self, nIter=10000):
        start_time = timeit.default_timer()
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y,
                   self.x_b_1_tf: self.x_b_1, self.y_b_1_tf: self.y_b_1,
                   self.x_b_2_tf: self.x_b_2, self.y_b_2_tf: self.y_b_2,
                   self.x_b_3_tf: self.x_b_3, self.y_b_3_tf: self.y_b_3,
                   self.x_b_4_tf: self.x_b_4, self.y_b_4_tf: self.y_b_4
                    }
        for it in range(nIter):


            
            self.sess.run(self.train_op, tf_dict)
            
            
            if it % 20 == 0:
                elapsed = timeit.default_timer() - start_time

                loss_c_value, loss_f_value = self.sess.run([self.loss_c, self.loss_f], tf_dict)

                self.loss_c_log.append(loss_c_value)
                self.loss_f_log.append(loss_f_value)


                print('It: %d, Loss_c: %.3e, Loss_f: %.3e, Time: %.2f' %
                      (it, loss_c_value, loss_f_value, elapsed))
                start_time = timeit.default_timer()




    def callback(self, loss):
        print('Loss: %e' % (loss))



    def train_p(self):
                
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y,
                   self.x_b_1_tf: self.x_b_1, self.y_b_1_tf: self.y_b_1,
                   self.x_b_2_tf: self.x_b_2, self.y_b_2_tf: self.y_b_2,
                   self.x_b_3_tf: self.x_b_3, self.y_b_3_tf: self.y_b_3,
                   self.x_b_4_tf: self.x_b_4, self.y_b_4_tf: self.y_b_4
                    }
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
        


   
    def predict_u(self, x, y):
        tf_dict = {self.x_tf: x, self.y_tf: y}  
        u = self.sess.run(self.u_pred, tf_dict)
        return u

    def predict_a(self, x, y):
        tf_dict = {self.x_tf: x, self.y_tf: y}  
        a = self.sess.run(self.a_pred, tf_dict)
        return a

    def predict_k(self, x, y):
        tf_dict = {self.x_tf: x, self.y_tf: y}  
        k = self.sess.run(self.k_pred, tf_dict)
        return k

    def predict_M(self, x, y):
        tf_dict = {self.x_tf: x, self.y_tf: y}  
        M = self.sess.run(self.M_pred, tf_dict)
        return M



def make_data_b(n=11):
    a = np.linspace(0, 1, n)
    a = a.reshape(a.shape[0], 1)
    b = np.zeros((n, 1))
    c = np.ones((n, 1))
    d1 = np.append(a, b, 1)
    d2 = np.append(a, c, 1)
    d3 = np.append(b, a, 1)
    d4 = np.append(c, a, 1)
    return d1, d2, d3, d4
    
def make_data_all(n=11):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    X = X.reshape(X.shape[0]*X.shape[1], 1)
    Y = Y.reshape(Y.shape[0]*Y.shape[1], 1)
    dian = np.append(X, Y, 1)
    return dian    





if __name__ == '__main__':

    layers = [[2]+3*[30]+[1], [2]+3*[20]+[3], [2]+3*[20]+[3]]

    
    config = {
        "font.family": 'serif',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
     }
    rcParams.update(config)

    X_b1_star, X_b2_star, X_b3_star, X_b4_star = make_data_b(n=31)
    X_b_star = [X_b1_star, X_b2_star, X_b3_star, X_b4_star]
    X_star = make_data_all(n=31)


    '''
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(X_b1_star[:,0], X_b1_star[:,1], 'v', markersize=3, label='Boundary1')
    plt.plot(X_b2_star[:,0], X_b2_star[:,1], '^', markersize=3, label='Boundary2')
    plt.plot(X_b3_star[:,0], X_b3_star[:,1], '<', markersize=3, label='Boundary3')
    plt.plot(X_b4_star[:,0], X_b4_star[:,1], '>', markersize=3, label='Boundary4')
    plt.plot(X_star[:,0], X_star[:,1], 'o', markersize=3, zorder=1, label='Random data')
    plt.legend(frameon=True, loc='upper right', labelspacing=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    '''


    N_data = 2000
    N_b1, N_b2, N_b3, N_b4 = 40, 40, 40, 40 

    idx_data = np.random.choice(X_star.shape[0], N_data)
    idx_b1 = np.random.choice(X_b1_star.shape[0], N_b1)
    idx_b2 = np.random.choice(X_b2_star.shape[0], N_b2)
    idx_b3 = np.random.choice(X_b3_star.shape[0], N_b3)
    idx_b4 = np.random.choice(X_b4_star.shape[0], N_b4)

    X_train = X_star[idx_data, :]
    X_b1_train ,X_b2_train, X_b3_train, X_b4_train = X_b1_star[idx_b1,:], X_b2_star[idx_b2,:], X_b3_star[idx_b3,:], X_b4_star[idx_b4,:]



    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(X_b1_train[:,0], X_b1_train[:,1], 'v', markersize=3, label='Boundary1')
    plt.plot(X_b2_train[:,0], X_b2_train[:,1], '^', markersize=3, label='Boundary2')
    plt.plot(X_b3_train[:,0], X_b3_train[:,1], '<', markersize=3, label='Boundary3')
    plt.plot(X_b4_train[:,0], X_b4_train[:,1], '>', markersize=3, label='Boundary4')
    plt.plot(X_train[:,0], X_train[:,1], 'o', markersize=3, zorder=1, label='Random data')
    plt.legend(frameon=True, loc='upper right', labelspacing=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



    model = PINN_model(layers, X_star, X_b_star)


    #model.train_p()

    model.train(100000)


    
    x = np.linspace(0, 1, 51)
    y = np.linspace(0, 1, 51)
    x, y = np.meshgrid(x, y)
    X_ = x.reshape(x.shape[0]*x.shape[1], 1)
    Y_ = y.reshape(y.shape[0]*y.shape[1], 1)
    dian = np.append(X_, Y_, 1)
    u_pred = model.predict_u(dian[:, 0:1], dian[:, 1:2])   
    u_pred_ = u_pred.reshape(51, 51)
        
    data = xlrd.open_workbook('shell_data_0.xls')
    table = data.sheets()[1]
    u_fem = np.array(table.col_values(0))
    u_fem_ = u_fem.reshape(51, 51)





    fig_1 = plt.figure(1, figsize=(20, 5), dpi=1000)
    plt.subplot(1, 3, 1)
    plt.contourf(x, y, u_pred_, levels=np.linspace(-0.042, 0.001, 30), cmap=plt.get_cmap('Spectral'))
    plt.colorbar(format='%.3f').set_label(label='u(x, y)', loc='center', size=15)
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    plt.title('PINN', size=17, weight='bold')
    
    plt.subplot(1, 3, 2)
    plt.contourf(x, y, u_fem_, levels=np.linspace(-0.042, 0.001, 30), cmap=plt.get_cmap('Spectral'))
    plt.colorbar(format='%.3f').set_label(label='u(x, y)', loc='center', size=15)
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    plt.title('FEM', size=17, weight='bold')

    error = u_pred_-u_fem_
    plt.subplot(1, 3, 3)
    plt.contourf(x, y, error, levels=np.linspace(np.amin(error),np.amax(error),30), cmap=plt.get_cmap('Spectral'))
    plt.colorbar(format='%.5f').set_label(label='u(x, y)', loc='center', size=15)   
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    plt.title('Error', size=17, weight='bold')    
    plt.show()




    #3D
    fig_3D=plt.figure(1, figsize=(5, 4), dpi=1000)
    ax=Axes3D(fig_3D)
    ax.plot_surface(x, y, u_pred_, rstride=1,cstride=1,cmap=plt.get_cmap('Spectral'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-0.06, 0.04)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    ax.set_title('u(x, y)  predicted by PINN')


    fig_3D=plt.figure(1, figsize=(5, 4), dpi=1000)
    ax=Axes3D(fig_3D)
    ax.plot_surface(x, y, u_fem_, rstride=1,cstride=1,cmap=plt.get_cmap('Spectral'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-0.06, 0.04)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    ax.set_title('u(x, y)  predicted by FEM')


    #  x
    fig_x = plt.figure(1, figsize=(10, 2.5), dpi=600)
    plt.plot(x[0], u_fem_[15], '-', label='FEM, y=0.3', linewidth=3.0, zorder=1, c='#FD841F')
    plt.plot(x[0], u_pred_[15], '^-',  label='PINN, y=0.3', linewidth=1.0, markersize=5, c='#3A8891')
    plt.plot(x[0], u_fem_[25], '--', label='FEM, y=0.5', linewidth=3.0, zorder=1, c='#3E6D9C')
    plt.plot(x[0], u_pred_[25], 'o-',  label='PINN, y=0.5', linewidth=1.0, markersize=5, c='#E14D2A')
    #plt.ylim(-0.05, 0.005)
    #plt.xlim(-0.1, 1.1)
    plt.xlabel('x', size=14, weight='bold')
    plt.ylabel('u(x, y)', size=14, weight='bold')
    plt.legend(frameon=False, fontsize=12)
    plt.show()


    #  y
    fig_x = plt.figure(1, figsize=(10, 2.5), dpi=600)
    #plt.scatter([0, 1], [0, 0], marker='o', s=100, c='#2878b5')
    plt.plot(x[0], u_fem_[:,15], '-', label='FEM, x=0.3', linewidth=3.0, zorder=1, c='#FD841F')
    plt.plot(x[0], u_pred_[:,15], '^-',  label='PINN, x=0.3', linewidth=1.0, markersize=5, c='#3A8891')
    plt.plot(x[0], u_fem_[:,25], '--', label='FEM, x=0.5', linewidth=3.0, zorder=1, c='#3E6D9C')
    plt.plot(x[0], u_pred_[:,25], 'o-',  label='PINN, x=0.5', linewidth=1.0, markersize=5, c='#E14D2A')
    #plt.ylim(-0.05, 0.005)
    #plt.xlim(-0.1, 1.1)
    plt.xlabel('y', size=14, weight='bold')
    plt.ylabel('u(x, y)', size=14, weight='bold')
    plt.legend(frameon=False, fontsize=12)
    plt.show()




    #loss
    fig_loss = plt.figure(figsize=(10, 4), dpi=300)
    ax = fig_loss.add_subplot(1, 2, 1)
    ax.plot(np.array(model.loss_f_log), c='#FA7F6F')
    ax.set_yscale('log')
    ax.set_xlabel('iterations')
    ax.set_ylabel('Loss')
    ax.set_title('loss_f')
    ax1 = fig_loss.add_subplot(1, 2, 2)
    ax1.plot(np.array(model.loss_c_log))
    ax1.set_yscale('log')
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title('loss_c')
    fig_loss.show()




