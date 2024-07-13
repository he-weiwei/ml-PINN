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


# PINN model
class PINN_model:
    def __init__(self, layers, X, X_u, X_M):
        
        self.max_X, self.min_X = X.max(0), X.min(0)


        self.x, self.y = X[:, 0:1], X[:, 1:2]
        self.x_u, self.y_u = X_u[:, 0:1], X_u[:, 1:2]
        self.x_M, self.y_M = X_M[:, 0:1], X_M[:, 1:2]

        self.q = -10.0

        
        self.v = 0.0
        self.E = 12000000.0
        self.t = 0.01
        self.D0 = (self.E*self.t**3) / (12*(1-self.v**2))
        self.D = self.D0 * tf.constant([[   1.0, self.v,            0.0],
                                        [self.v,    1.0,            0.0],
                                        [   0.0,    0.0, (1-self.v)/2.0]])
        
        
        
        self.layers_u, self.layers_k, self.layers_M = layers[0], layers[1], layers[2]
        self.weights_u, self.biases_u = self.initialize_NN(self.layers_u)
        self.weights_k, self.biases_k = self.initialize_NN(self.layers_k)
        self.weights_M, self.biases_M = self.initialize_NN(self.layers_M)
        

        
        
        self.x_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.y_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.x_u_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.y_u_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.x_M_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.y_M_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])


        
        self.u_pred = self.net_u(self.x_tf, self.y_tf)
        self.k_pred = self.net_k(self.x_tf, self.y_tf)
        self.M_pred = self.net_M(self.x_tf, self.y_tf)
        
        self.u_c_pred = self.net_u(self.x_u_tf, self.y_u_tf)
        self.M_c_pred = self.net_M(self.x_M_tf, self.y_M_tf)

        self.f_u_k_pred, self.f_k_M_pred, self.f_M_q_pred = self.net_f(self.x_tf, self.y_tf)
        
        
        self.loss_c = 1e3*tf.reduce_mean(tf.square(self.u_c_pred)) \
            + tf.reduce_mean(tf.square(self.M_c_pred[:, 0:3])) \


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
                    self.x_u_tf: self.x_u, self.y_u_tf: self.y_u,
                    self.x_M_tf: self.x_M, self.y_M_tf: self.y_M,
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
                    self.x_u_tf: self.x_u, self.y_u_tf: self.y_u,
                    self.x_M_tf: self.x_M, self.y_M_tf: self.y_M,
                    }
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
        


    
    def predict_u(self, x, y):
        tf_dict = {self.x_tf: x, self.y_tf: y}  
        u = self.sess.run(self.u_pred, tf_dict)
        return u

    def predict_k(self, x, y):
        tf_dict = {self.x_tf: x, self.y_tf: y}  
        k = self.sess.run(self.k_pred, tf_dict)
        return k

    def predict_M(self, x, y):
        tf_dict = {self.x_tf: x, self.y_tf: y}  
        M = self.sess.run(self.M_pred, tf_dict)
        return M






def make_X(n=21):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    X = X.reshape(X.shape[0]*X.shape[1], 1)
    Y = Y.reshape(Y.shape[0]*Y.shape[1], 1)
    dian = np.append(X, Y, 1)
    return dian

def make_Xc(n=21):
    a = np.linspace(0, 1, n)
    a = a.reshape(a.shape[0], 1)
    b = np.zeros((n, 1))
    c = np.ones((n, 1))
    d1 = np.append(a, b, 1)
    d2 = np.append(a, c, 1)
    d3 = np.append(b, a, 1)
    d4 = np.append(c, a, 1)
    return np.vstack((d3, d4))
    
    


if __name__ == '__main__':

    layers = [[2]+4*[30]+[1], [2]+4*[20]+[3], [2]+4*[20]+[3]]

    
    config = {
        "font.family": 'serif',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
     }
    rcParams.update(config)


    n = 21
    X_star = make_X(n)
    X_u_star = make_Xc(n)
    X_M_star = X_u_star    

    N_1 = 1000
    idx_1 = np.random.choice(X_star.shape[0], N_1)
    X_train = X_star[idx_1, :]
    
    N_2 = 300
    idx_2 = np.random.choice(X_u_star.shape[0], N_2)
    X_u_train = X_u_star[idx_2, :]
    
    X_M_train = X_u_train
    

    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(X_train[:, 0:1], X_train[:, 1:2], s=10, marker='o', label='X')
    plt.scatter(X_u_train[:, 0:1], X_u_train[:, 1:2], s=5, marker='^', label='X_uc')
    plt.legend(frameon=True, loc='upper right', edgecolor='r',labelspacing=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()






    model = PINN_model(layers, X_star, X_u_star, X_M_star)

    model.train(100000)
    #model.train_p()


    
    x = np.linspace(0, 1, 51)
    y = np.linspace(0, 1, 51)
    x, y = np.meshgrid(x, y)
    X_ = x.reshape(x.shape[0]*x.shape[1], 1)
    Y_ = y.reshape(y.shape[0]*y.shape[1], 1)
    dian = np.append(X_, Y_, 1)
    u_pred = model.predict_u(dian[:, 0:1], dian[:, 1:2])
    u_pred_ = u_pred.reshape(51, 51)




    data = xlrd.open_workbook('shell_data_0.xls')
    table = data.sheets()[0]
    u_fem = np.array(table.col_values(0))
    u_fem_ = u_fem.reshape(51, 51)





    
    fig_1 = plt.figure(1, figsize=(20, 5), dpi=1000)
    plt.subplot(1, 3, 1)
    plt.contourf(x, y, u_pred_, levels=np.linspace(-0.14, 0.001,30), cmap=plt.get_cmap('Spectral'))
    plt.colorbar(format='%.2f').set_label(label='u(x, y)', loc='center', size=15)
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    plt.title('PINN', size=17, weight='bold')
    
    plt.subplot(1, 3, 2)
    plt.contourf(x, y, u_fem_, levels=np.linspace(-0.14, 0.001,30), cmap=plt.get_cmap('Spectral'))
    plt.colorbar(format='%.2f').set_label(label='u(x, y)', loc='center', size=15)
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    plt.title('FEM', size=17, weight='bold')
    
    plt.subplot(1, 3, 3)
    error = u_pred_-u_fem_
    plt.contourf(x, y, error, levels=np.linspace(np.amin(error),np.amax(error),30), cmap=plt.get_cmap('Spectral'))
    plt.colorbar(format='%.4f').set_label(label='u(x, y)', loc='center', size=15)   
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    plt.title('Error', size=17, weight='bold')    
    plt.show()



    #  x
    fig_x = plt.figure(1, figsize=(10, 2.5), dpi=600)
    plt.plot(x[0], u_pred_[25], 'o-',  label='PINN, y=0.5', linewidth=1.0, c='#E14D2A')
    plt.plot(x[0], u_fem_[25], label='FEM, y=0.5', linewidth=3.0, c='#3E6D9C', zorder=1)
    plt.xlabel('x', size=14, weight='bold')
    plt.ylabel('u(x, y)', size=14, weight='bold')
    plt.legend(frameon=False, fontsize=12)
    plt.show()


    #  y
    fig_x = plt.figure(1, figsize=(10, 2.5), dpi=600)
    #plt.scatter([0, 1], [0, 0], marker='o', s=100, c='#2878b5')
    plt.plot(x[0], u_fem_[:,10], '-', label='FEM, x=0.2', linewidth=3.0, zorder=1, c='#FD841F')
    plt.plot(x[0], u_pred_[:,10], '^-',  label='PINN, x=0.2', linewidth=1.0, markersize=5, c='#3A8891')
    plt.plot(x[0], u_fem_[:,25], '--', label='FEM, x=0.5', linewidth=3.0, zorder=1, c='#3E6D9C')
    plt.plot(x[0], u_pred_[:,25], 'o-',  label='PINN, x=0.5', linewidth=1.0, markersize=5, c='#E14D2A')
    plt.ylim(-0.14, 0.1)
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
    #ax1 = fig_loss.add_subplot(1, 2, 2)
    ax.plot(np.array(model.loss_c_log))
    ax.set_yscale('log')
    ax.set_xlabel('iterations')
    ax.set_ylabel('Loss')
    ax.set_title('loss_c')
    fig_loss.show()


