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
    def __init__(self, layers, X, q, X_uc, u_c, X_ac, a_c, X_kc, k_c):
        
        self.max_X, self.min_X = X.max(0), X.min(0)

        self.layers_u = layers[0]
        self.layers_a = layers[1]
        self.layers_k = layers[2]
        self.layers_Q = layers[3]


        self.X = X
        self.X_uc, self.u_c = X_uc, u_c
        self.X_ac, self.a_c = X_ac, a_c
        self.X_kc, self.k_c = X_kc, k_c
        self.q = q        

        
        
        self.EI = 1.0
        
        
        
        self.weights_u, self.biases_u = self.initialize_NN(self.layers_u)
        self.weights_a, self.biases_a = self.initialize_NN(self.layers_a)
        self.weights_k, self.biases_k = self.initialize_NN(self.layers_k)
        self.weights_Q, self.biases_Q = self.initialize_NN(self.layers_Q)
        

        
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        
        
        
        self.x_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.q_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.x_u_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.x_a_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.x_k_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.u_c_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.a_c_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.k_c_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        
        
        
        self.u_pred = self.net_u(self.x_tf)
        self.a_pred = self.net_a(self.x_tf)
        self.k_pred = self.net_k(self.x_tf)
        self.Q_pred = self.net_Q(self.x_tf)
        
        
        self.u_c_pred = self.net_u(self.x_u_tf)
        self.a_c_pred = self.net_a(self.x_a_tf)
        self.k_c_pred = self.net_k(self.x_k_tf)
        

        self.f_Q_q_pred, self.f_M_Q_pred, self.f_a_k_pred, self.f_u_k_pred = self.net_f(self.x_tf)        
        
        

        

        self.loss_c = tf.reduce_mean(tf.square(self.u_c_pred - self.u_c_tf)) \
            + tf.reduce_mean(tf.square(self.k_c_pred - self.k_c_tf))

        

        self.loss_f = tf.reduce_mean(tf.square(self.f_Q_q_pred)) \
            + tf.reduce_mean(tf.square(self.f_M_Q_pred)) \
            + tf.reduce_mean(tf.square(self.f_a_k_pred)) \
            + tf.reduce_mean(tf.square(self.f_u_k_pred))


        
        self.loss = self.loss_c +  self.loss_f
          
        
        
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 5e-4
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
        

        
        self.loss_log = []
        self.loss_c_log = []
        self.loss_f_log = []
        self.saver = tf.train.Saver()

        
        init = tf.global_variables_initializer()
        self.sess.run(init)


    
    # Xavier initialization
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



    

    def net_u(self, x):
        u = self.forward_pass(tf.concat([x], 1), self.weights_u, self.biases_u, self.layers_u)
        return u
    
    def net_a(self, x):
        a = self.forward_pass(tf.concat([x], 1), self.weights_a, self.biases_a, self.layers_a)
        return a
    
    def net_k(self, x):
        k = self.forward_pass(tf.concat([x], 1), self.weights_k, self.biases_k, self.layers_k)
        return k
    
    def net_Q(self, x):
        Q = self.forward_pass(tf.concat([x], 1), self.weights_Q, self.biases_Q, self.layers_Q)
        return Q
    



    def net_f(self, x):
        Q, k, a, u = self.net_Q(x), self.net_k(x), self.net_a(x), self.net_u(x)
        M = self.EI * k
        Q_x = tf.gradients(Q, x)[0]
        M_x = tf.gradients(M, x)[0]
        a_x = tf.gradients(a, x)[0]
        u_x = tf.gradients(u, x)[0]
        
        f_Q_q = Q_x + self.q_tf
        f_M_Q = M_x - Q
        f_a_k = a_x + k
        f_u_k = u_x - a
        
        return f_Q_q, f_M_Q, f_a_k, f_u_k





    
    def train(self, nIter=10000):
        start_time = timeit.default_timer()
        tf_dict = {self.x_u_tf: self.X_uc, self.u_c_tf: self.u_c,
                    self.x_a_tf: self.X_ac, self.a_c_tf: self.a_c,
                    self.x_k_tf: self.X_kc, self.k_c_tf: self.k_c,
                    self.x_tf: self.X,
                    self.q_tf: self.q
                    }
        for it in range(nIter):

            
            
            self.sess.run(self.train_op, tf_dict)
            
            
            
            elapsed = timeit.default_timer() - start_time

            loss_value, loss_c_value, loss_f_value = self.sess.run([self.loss, self.loss_c, self.loss_f], tf_dict)
            self.loss_log.append(loss_value)
            self.loss_c_log.append(loss_c_value)
            self.loss_f_log.append(loss_f_value)


            if it % 100 == 0:
                print('It: %d, Loss_c: %.3e, Loss_f: %.3e, Time: %.2f' %
                      (it, loss_c_value, loss_f_value, elapsed))
                start_time = timeit.default_timer()




    def callback(self, loss):
        print('Loss: %e' % (loss))



    def train_p(self):
                
        tf_dict = {self.x_u_tf: self.X_uc, self.u_c_tf: self.u_c,
                    self.x_a_tf: self.X_ac, self.a_c_tf: self.a_c,
                    self.x_k_tf: self.X_kc, self.k_c_tf: self.k_c,
                    self.x_tf: self.X,
                    self.q_tf: self.q
                    }
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
        


    
    def predict_u(self, x):
        tf_dict = {self.x_tf: x} 
        u = self.sess.run(self.u_pred, tf_dict)
        return u
    
    def predict_a(self, x):
        tf_dict = {self.x_tf: x} 
        a = self.sess.run(self.a_pred, tf_dict)
        return a

    def predict_k(self, x):
        tf_dict = {self.x_tf: x} 
        k = self.sess.run(self.k_pred, tf_dict)
        return k

    def predict_Q(self, x):
        tf_dict = {self.x_tf: x} 
        Q = self.sess.run(self.Q_pred, tf_dict)
        return Q





if __name__ == '__main__':

    layers = [[1]+3*[10]+[1], [1]+3*[10]+[1], [1]+3*[10]+[1], [1]+3*[10]+[1]]

    data = xlrd.open_workbook('beam_data.xls')
    table = data.sheets()[0]
    x_fem = np.array(table.col_values(0))
    u_fem = np.array(table.col_values(1))


    X_star = np.linspace(0, 1, 1001)
    X_star.resize((X_star.shape[0], 1))
    N_f = 100
    idx = np.random.choice(X_star.shape[0], N_f)
    X_train = X_star[idx, :]



    plt.figure(figsize=(10, 1), dpi=300)
    plt.plot([0, 1], [0, 0], '^', markersize=10, label='Boundary')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.7, 2.0)
    plt.plot(X_train, np.zeros(X_train.shape), 'o', markersize=8, alpha=0.5, label='Random data')
    plt.yticks([])
    plt.xlabel('Location  x')
    plt.legend(fontsize=12, ncol=2)
    plt.show()

    
    q = np.zeros(X_train.shape) 

    
    q[:] = -1.0
    X_uc = np.array([[0.0], [1.0]])
    u_c = np.array([[0.0], [0.0]])
    
    X_ac = np.array([[0.0], [1.0]])
    a_c = np.array([[0.0], [0.0]])
    
    X_kc = np.array([[0.0], [1.0]])
    k_c = np.array([[0.0], [0.0]])


    model = PINN_model(layers, X_train, q, X_uc, u_c, X_ac, a_c, X_kc, k_c)

    
    model.train(10000)



    
    config = {
        "font.family": 'serif',
        "font.size": 16,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
     }
    rcParams.update(config)

    Q_pred, k_pred, a_pred, u_pred = model.predict_Q(X_star), model.predict_k(X_star), model.predict_a(X_star), model.predict_u(X_star)



    plt.figure(figsize=(7.5, 1.5), dpi=300)
    plt.scatter([0.0, 1.0], [0.0, 0.0], marker='^', s=100, c='#65647C')
    plt.plot(x_fem, u_fem, label='FEM', linewidth=3.0, c='#2878b5', zorder=1)
    plt.plot(X_star[::25, :], u_pred[::25, :], '--',  label='PINN', linewidth=3.0, c='#cd5334')
    #plt.plot(X_train, np.zeros(X_train.shape), 'o', alpha=0.6, label='Nodes, num='+str(N_f), c='#FD841F')
    #plt.ylim(-0.04, 0.03)
    plt.xlabel(r'$x/l$')
    plt.ylabel(r'$\omega(x)$')
    #plt.xticks([])
    #plt.yticks([])
    plt.title('')
    plt.legend(frameon=False, fontsize=12)
    plt.show()


    def error(u_fem, u_pred):
        u_pred = u_pred[::10, :].reshape((101,))
        error = u_fem - u_pred
        return np.linalg.norm(error)/np.linalg.norm(u_fem)
    
    error = error(u_fem, u_pred)



    from matplotlib.ticker import FuncFormatter
    plt.figure(figsize=(6, 4), dpi=500)
    #plt.plot(np.array(model.loss_log), label='Loss$_t$')
    plt.plot(np.array(model.loss_f_log), '-', label='Loss$_p$')
    plt.plot(np.array(model.loss_c_log), '-',label='Loss$_b$')

    plt.ylim(1e-11, 1e1)
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend(frameon=False, fontsize=15)
    plt.show()

    
    print(np.array(model.loss_f_log)[-1]/N_f)
    print(np.array(model.loss_c_log)[-1])
    print(error)
    








