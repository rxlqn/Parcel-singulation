###### 控制文件
## 包含观测（从仿真文件采样或者从现实世界获取数据），预测（基于模型或者历史经验进行预测），决策部分

import numpy as np
import time


# MailCoordinate *：输入的定位信息
# BeltVelocity *：输出定位结果


def Control(object):        # 封装好的控制函数

    start = time.time()
    sample_pixel = Sample(object,object.Parcels)
    end = time.time()
    # print("采样时间:",str(end-start))

    Control_Policy(object,sample_pixel)

def cal_policy():           # 计算控制策略
    pass

def belt_control():         # 根据计算出的策略进行控制
    pass


def Sample(self,Parcels):   # 分析采样图像后得到的包裹信息
    # 更新pixel
    parcels_pixel = []

    for index in range(0, len(Parcels)):         # 存在先后问题，不能直接删除
        try:
            parcel = Parcels[index]
        except:
            print()
        sub = 4         # 计算控制策略可以降采样
        pixel = self.Generate_pixel(sub, index)
        parcels_pixel.append(pixel)
    return parcels_pixel       


def Control_Policy(self, parcels_pixel):          # 需要一个实时优先级变化
    

    start = time.time()

    # 计算占用传送带的情况
    parcels_num_act = []
    unsort_prio = []
    for pix in parcels_pixel:          

        num_act = self.cal_num_act(pix)
        parcels_num_act.append(num_act)
        unsort_prio.append(pix[:,0].max())

    end = time.time()

    # print("控制策略传送带占用情况:",str(end-start))

    # 计算当前包裹的优先级
    sort_prio = np.sort(-np.array(unsort_prio))      # 降序排列

    for index in range(0, len(self.Parcels)):            # 每个包裹的最大x值在当前包裹中的顺序作为优先级
        self.Parcels[index].prio = np.where(np.sort(-np.array(unsort_prio))==-unsort_prio[index])[0][0]
    
    # 控制只控制右方传送带，避免将后方的包裹带起同时运动
    # 根据相邻两个包裹之间到达D线的时间差去设定速度差 
    sort_index = np.argsort(-np.array(unsort_prio))  # 降序下标




    start = time.time()


    # 计算每个包裹需要的速度，可以迭代去逼近            #迭代可能需要计算上一个传送带到达终点的时间
    max_speed = 40
    parcel_speed = [max_speed]
    D = 520
    delta_T = 5
    T0 = (D - self.Parcels[sort_index[0]].x)/max_speed
    for i in range(1,len(sort_index)):
        Ti = T0 + delta_T*i
        index = sort_index[i]
        parcel_speed.append(int((D-self.Parcels[index].x)/Ti))

    sort_index = np.argsort(np.array(unsort_prio))  # 升序下标

    # 按照实时优先级控制速度    
    for index,i in enumerate(sort_index):        # 先访问优先级低的
        num_act = parcels_num_act[i]
        uniques = np.unique(num_act)
        # speed = 10/len(parcels_num_act)*(index+1)

        # max_s = 100
        # delta_s = 20
        # speed = max_s+delta_s-delta_s*len(parcels_num_act)+index*delta_s
        speed = parcel_speed[len(parcels_num_act)-1-index]    # 优先级相反，先赋值优先级最低得
        if speed <= 0:
            speed = 0
        
        num = []
        for t in uniques:
            if t!=-1:
                num.append(t)
        res =  np.array(num)%self.act_array.col                                                            ###################################

        for i,t in enumerate(res):          # 最后一排传送带会整除
            if t == 0 and num[i]!=0:
                res[i] = self.act_array.col      
        res_max = res.max()

        for i in range(0,len(num)):
            if res[int(i)] == res_max:
                index = int(i)
                self.act_array.actuator[int(num[index])].speed = speed

        # for i in uniques:
        #     if i != -1:
        #         index = int(i)
        #         self.act_array.actuator[index].speed = speed

    end = time.time()
    # print("控制策略速度计算时间:",str(end-start))

def cal_priority(self):
    pass
