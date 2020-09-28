重构代码，仿真部分自定义游戏规则(不包含旋转)，降采样分辨率
window.py 渲染界面，定时运行仿真模型
simulation.py 仿真文件
actuator_array 执行器阵列预定义文件

control_policy.py 人为控制策略，后面用强化学习模型替代

AC 模型训练两个并列包裹，控制输出距离。
需要设计reward 

每个传送带为一个agent，分别构建神经网络，共享一个critic
actor 模型 *17
input 2*6  
output 10 级可调 


reward 过终点线记录包裹之间的间距，超过dis的距离reward指数降低，小于dis惩罚,未过线前均为-1(希望尽快抵达终点)
state 当前包裹位置信息 dim 3  [坐标x,坐标y, l,w,v, a] (先测试2个包裹)  
input state 需要batch normalization 保证数据处于相同的scale

action 传送带 dim 2 [传送带index,velocity] 21  (21个传送带)    总的情况可能很多，维度2
        21个传送带，从5级速度中choose速度

$ source /d/Program/Anaconda/etc/profile.d/conda.sh
tensorboard --port 6006 --logdir logs

todo:
一个包裹，强化学习到终点
两个并列包裹，加上强化学习策略，控制输出距离，顺便重写代码
多个包裹，加上碰撞后控制到终点的时间

todo:
1、加入碰撞
2、优化代码
3、设计reward方式
4、使用actor critic方式代替人为决策

actor critic advantage 版本收敛效果不太好

85个输出时并不收敛