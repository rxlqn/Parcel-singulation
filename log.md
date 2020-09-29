### DDPG

#### Exp 1

- MAX_EPISODES = 500

  MAX_EP_STEPS = 1000

  MEMORY_CAPACITY = 100000

  BATCH_SIZE = 64

  LR_A = 0.0001  *# learning rate for actor*

  LR_C = 0.001  *# learning rate for critic*

  GAMMA = 0.99   *# reward discount*

  TAU = 0.001   *# soft replacement*

  MEMORY_CAPACITY = 100000

  BATCH_SIZE = 64

- ```python
      def _build_a(self, s, scope, trainable):
          with tf.variable_scope(scope):
              net1 = tf.layers.dense(s, 400, activation=tf.nn.relu, name='l1', trainable=trainable)
              net2 = tf.layers.dense(net1, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
  
              a = tf.layers.dense(net2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
              a = tf.add(a,1)
              return tf.multiply(a, self.a_bound, name='scaled_a')
  
      def _build_c(self, s, a, scope, trainable):
          with tf.variable_scope(scope):
              l1 = tf.layers.dense(s, 400, activation=tf.nn.relu, name='l1', trainable=trainable)
  
              n_l2 = 300
              w2_s = tf.get_variable('w2_s', [400, n_l2], trainable=trainable)
              w2_a = tf.get_variable('w2_a', [self.a_dim, n_l2], trainable=trainable)
              b2 = tf.get_variable('b2', [1, n_l2], trainable=trainable)
              net = tf.nn.relu(tf.matmul(l1, w2_s) + tf.matmul(a, w2_a) + b2)
              return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
  ```


- ![image-20200927225320693](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20200927225320693.png)
- ![image-20200928151932272](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20200928151932272.png)
- ![image-20200928151951399](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20200928151951399.png)
- ![image-20200927231002187](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20200927231002187.png)
- ![image-20200927231014325](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20200927231014325.png)

#### Exp 2

- add BN（not work)

#### Exp 3

- add done_flag
- normalize gradients(according to papper)
- same hyperparameter
- have better ep_reward
- <img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20200929131419166.png" alt="image-20200929131419166" style="zoom: 67%;" />
- <img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20200929131433509.png" alt="image-20200929131433509" style="zoom:67%;" />