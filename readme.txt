重构代码，仿真部分自定义游戏规则(不包含旋转)，降采样分辨率
window.py 渲染界面，定时运行仿真模型
simulation.py 仿真文件
actuator_array 执行器阵列预定义文件

control_policy.py 人为控制策略，后面用强化学习模型替代

AC 模型训练两个并列包裹，控制输出距离。
需要设计reward 
过终点线记录包裹之间的间距，超过dis的距离reward指数降低，小于dis惩罚



$ source /d/Program/Anaconda/etc/profile.d/conda.sh

todo:
一个包裹，强化学习到终点
两个并列包裹，加上强化学习策略，控制输出距离，顺便重写代码
多个包裹，加上碰撞后控制到终点的时间

todo:
1、加入碰撞
2、优化代码
3、设计reward方式
4、使用actor critic方式代替人为决策


