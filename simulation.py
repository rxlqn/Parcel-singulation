###### 仿真建模文件
## 包含像素级分析的包裹生成，每个包裹计算合力，合力矩，分析下一时间的受力情况
from math import pi, cos, sin, floor
import sys
import matplotlib.pyplot as plt
import numpy as np
import actuator_array
import time
import copy
import actuator_array as act

Pixel_array = []
T = 0

class Physic_simulation():

    def __init__(self): 
        self.Parcels = []           # 生成包裹需要考虑当前的像素占用情况
        self.act_array = act.Actuator_array() 


    # 根据当前包裹位置生成仿真所需要的像素点
    def Generate_pixel(self, subsample, index):

        parcel = self.Parcels[index]
        theta = parcel.theta/180*pi
        M = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

        pixel = np.ones(
            [(parcel.l+1)//subsample, (parcel.w+1)//subsample, 2])   # 生成包裹矩阵

        for i in range(0, (parcel.l+1)//subsample):
            for j in range(0, (parcel.w+1)//subsample):
                pixel[i][j] = ([parcel.x-(parcel.l-1)/2+i*subsample,
                                parcel.y - (parcel.w-1)/2+j*subsample])

        pixel.resize((parcel.w+1)//subsample*(parcel.l+1) //
                     subsample, 2)       # 拉成一维
        pixel = np.dot(pixel-parcel.r_cm, M)+parcel.r_cm       # 旋转后的点
        return pixel

    # 用矩阵计算代替循环优化计算速度,经过分析，np.array占用大量时间，大致为0.14-0.18不可以在循环内进行转换
    def cal_Force(self, pixel, num_act, index):
        # 分与不同的传送带上的接触面积去计算受力情况
        F = np.array([0, 0])
        parcel = self.Parcels[index]
        F_test = []
        for i in range(0, num_act.size):
            if num_act[i] == -1:
                speed = 0
            else:
                speed = self.act_array.actuator[int(num_act[i])].speed
            F = F + self.cal_force(speed, pixel[i], parcel.r_cm,
                                   parcel.v_cm, parcel.omega, parcel.mu, parcel.k, 1)
            # F_test.append(self.cal_force(self.act_array.actuator[int(num_act[i])].speed,pixel[i],1))
        F[1] = 0
        return F

    # 力矩产生平面上的旋转
    def cal_Torque(self, pixel, num_act, index):
        torque = 0
        parcel = self.Parcels[index]
        R = []
        f = []
        for i in range(0, num_act.size):
            R.append(pixel[i]-parcel.r_cm)
            if num_act[i] == -1:
                speed = 0
            else:
                speed = self.act_array.actuator[int(num_act[i])].speed
            f.append(self.cal_force(
                speed, pixel[i], parcel.r_cm, parcel.v_cm, parcel.omega, parcel.mu, parcel.k, 2))
        # 首先要保证受力后的力矩为0，主要考虑边界上，可能会误分,piexel无法做到完全的对称。。。艹烦躁
        # 减小因为pixel带来的转动惯量的影响，1增大pixel数量，2减小转动惯量的大小，这样较小的转动惯量对于整体的影响很小
        # 加速度和角加速度均与物体质量无关，所以还是需要增大接触面积
        torque = np.cross(R, f)          # 一次大量计算比多次少量计算快很多
        torque = sum(torque)
        return torque

    # 执行器速度，质点位置，质心位置，质心速度，角速度，mu,k
    def cal_force(self, act_speed, r, r_cm, V_cm, omega, mu, k, flag):

        # r_cm = parcel.r_cm
        U_r = [act_speed, 0]      # 质点处传送带的速度       120-300mm/s
        # V_cm = parcel.v_cm
        # omega = parcel.omega
        rotate = [[0, 1], [-1, 0]]

        R = np.dot([r[0]-r_cm[0], r[1]-r_cm[1]], rotate)
        Dv = 20                            # 振荡上升表面分界值太大或者力力矩太大,正负振荡表明分界值太小
        Rv = 5
        fr = 0
        if flag == 1:       # 计算合力      ,假设速度大于Dv时与速度无关，小于Dv时与速度线性，Dv可以修正
            if U_r[0] - V_cm[0] > -Dv and U_r[0] - V_cm[0] < Dv:      # 相对速度差比较
                fr = mu*k*(U_r - V_cm)/Dv
            else:
                fr = mu*k*(U_r - V_cm)/np.linalg.norm(abs((U_r - V_cm)))

        else:               # 计算合力矩，质心处速度的合力矩为0，所以只考虑后半部分,omega逆时针为正所以速度方向与U_R相反
            if np.linalg.norm(U_r - V_cm + omega*R) > -Rv and np.linalg.norm(U_r - V_cm + omega*R) < Rv:
                fr = mu*k*(U_r - V_cm + omega*R)/Rv
            else:
                fr = mu*k*(U_r - V_cm + omega*R) / \
                    np.linalg.norm(U_r - V_cm + omega*R)
        return fr

    # 判断质点r处于哪个执行器上
    def cal_num_act(self, pixel):
        # 超出边界的像素不参与计算
        bound = self.act_array.actuator[1].w*(self.act_array.col+1)+self.act_array.gap_w*(self.act_array.col) 
        for pix in pixel:
            if pix[0]>bound:
                pix[0] = bound
            if pix[0]<0:
                pix[0] = 0
        # pixel[pixel[:,0]>bound] = [bound,pixel[:,1]]
        # pixel[pixel[:,0]<0] = 0

        # 计算num_act矩阵
        quotient_x = pixel[:,
                           0]//(self.act_array.actuator[1].w+self.act_array.gap_w)  # 第几列
        remainder_x = pixel[:, 0] % (self.act_array.actuator[1].w+self.act_array.gap_w)

        # quotient_x[ quotient_x> self.act_array.actuator[1].w] = self.act_array.actuator[1].w      # 限制列

        quotient_y = pixel[:, 1]//(self.act_array.actuator[1].h +
                                   self.act_array.gap_h)    # 第几行
        remainder_y = pixel[:, 1] % (self.act_array.actuator[1].h+self.act_array.gap_h)

        # quotient_y[ quotient_y> self.act_array.actuator[1].h] = self.act_array.actuator[1].h      # 限制行

        num_act = quotient_y*self.act_array.col + quotient_x

        for i in range(0, num_act.size):
            if remainder_x[i] > self.act_array.actuator[1].w:  # 落在gap中，可以直接返回
                num_act[i] = -1
            if quotient_x[i] != 0 and remainder_y[i] > self.act_array.actuator[1].h:  # 落在gap中，直接返回
                num_act[i] = -1
            if quotient_x[i] == 0:
                num_act[i] = 0
            if num_act[i] < -1:  # 定上下限
                num_act[i] = 0
            elif num_act[i] > self.act_array.col*self.act_array.row:  # 0-32  超过上限不受力
                num_act[i] = -1

        return num_act

    def Parcel_sim(self):
 
        global T
        temp_time = 0
        T = T + 1
        parcels_temp = []
        # try:
        #     print("speed1:",self.Parcels[0].v_cm[0])
        #     print("speed2:",self.Parcels[1].v_cm[0])
        #     print("speed3:",self.Parcels[2].v_cm[0])
        #     print()
        # except:
        #     pass
        # 计算每个包裹的受力情况，后面可以改成矩阵运算加快计算速度
        for index in range(0, len(self.Parcels)):         # 存在先后问题，不能直接删除
            parcel = self.Parcels[index]
            sub = 8
            # subsample 注意原来的长宽高+1/subsample 需要为整数
            pixel = self.Generate_pixel(sub, index)
            num_act = self.cal_num_act(pixel)

            # #更新parcel中心点的位置

            start = time.time()
            F = self.cal_Force(pixel, num_act, index)*sub*sub       # 质量 受力

            end = time.time()
            temp_time = end-start+temp_time
            
            # print(F)
            parcel.a_cm = (F)/(parcel.m)
            # print(parcel.a_cm[0]/parcel.v_cm[0])   ### 大于-1,小于-1会产生振荡
            parcel_v_cm = parcel.v_cm + 1*parcel.a_cm
            try:
                # 一个单位时间更新一次，减小刷新步长
                parcel_x = parcel.x + floor(parcel_v_cm[0]*1)       
            except:
                print(1)

            # #更新parcel的旋转角度
            # torque = self.cal_Torque(pixel,num_act,index)/10*4
            # if torque != 0:
            #     print()
            # parcel.alpha = - torque/(parcel.J)        # 转动惯量有点略大
            # # print(parcel.alpha/parcel.omega)

            # parcel.omega = parcel.omega + 1*parcel.alpha #取整
            # parcel.theta = (parcel.theta + parcel.omega/pi*180)%360
            # print("omega:",round(parcel.omega/pi*180,1),"alpha:",round(parcel.alpha,1),"theta:",round(parcel.theta))
            # print()

            # 在计算完旋转角之后更新位移
            parcel.v_cm = parcel_v_cm
            parcel.x = parcel_x
            parcel.r_cm[0] = parcel_x

            # if parcel.x<500:    ## D线位置
            #     parcel.T  = parcel.T + 1

            # 判断越界，如果超出xg 将Parcel删除
            if(parcel.x < 520):
                parcels_temp.append(parcel)
                # parcel.T  = parcel.T + 1
            else:
                parcel.T = T + (parcel.x + (parcel.l-1)/2-520)/parcel.v_cm[0]     ## 实测时间
                print(parcel.T)


        # print("仿真受力计算时间",str(temp_time))
        self.Parcels = parcels_temp      # 删除越界的包裹

        # 更新pixel
        parcels_pixel = []

        for index in range(0, len(self.Parcels)):         # 存在先后问题，不能直接删除
            try:
                parcel = self.Parcels[index]
            except:
                print()
            sub = 1         # 生成包裹所需要的像素信息不能降采样
            pixel = self.Generate_pixel(sub, index)
            parcels_pixel.append(pixel)

        # 生成执行器1上的像素点矩阵
        w = self.act_array.actuator[0].w
        h = self.act_array.row * \
            self.act_array.actuator[0].h+(self.act_array.row-1)*self.act_array.gap_h

        act1_matrix = np.zeros([w+2, h+2])

        for parcel_pix in parcels_pixel:
            for pix in parcel_pix:
                x = pix[0]
                y = pix[1]
                if x>=0:
                    if x <= w and y<=h:  #必须取等号
                        act1_matrix[int(x), int(y)] = 1

        # # 根据位置执行策略,一个周期修改一次
        # self.Control_Policy(parcels_pixel)

        start = time.time()

        # 生成包裹
        try:
            last_parcel = self.Parcels[-1]       # 起始点为最后一个包裹的右下角的点
            start_x = min(int(last_parcel.x + (last_parcel.l-1)//2),w)
            start_y = int(last_parcel.y + (last_parcel.w-1)//2)     # 暂时没有使用

        except:
            start_x = w
            start_y = 0
        self.Generate_parcels(act1_matrix, start_x, start_y)

        end = time.time()
        # print("仿真包裹生成时间",str(end-start))



    def Generate_parcels(self, act1_matrix, start_x, start_y):

        w = self.act_array.actuator[0].w
        h = self.act_array.row * \
            self.act_array.actuator[0].h+(self.act_array.row-1)*self.act_array.gap_h

        exit_flag = 0
        insert_flag = 0
        # start = time.time()

        prob = np.random.randint(1,11)   #1到10随机数
        temp = actuator_array.Parcel(prob)

        for i in range(start_x, 10, -1):       # 从第一个执行器的右上角开始,找到第一个没有被占用的像素点
            for j in range(0, int(h), 1):
                exit_flag = 0

                # 如果没有被占用，遍历新temp包裹的四条边，如果没有像素点重复则可以插入
                if self.isoccupy([i, j], act1_matrix) == 0:



                    temp_l = temp.l
                    temp_w = temp.w
                    # temp_x = temp.x
                    # temp_y = temp.y
                    for t in range(0, temp_l+1):    # 遍历上边界
                        if i-t<0:       # 执行器左端的点不判断
                            break                    
                        if self.isoccupy([i-t, j], act1_matrix) == 1:
                            exit_flag = 1
                            break
                    if exit_flag == 1:      # 跳出当前循环
                        continue
                    for t in range(0, temp_w+1):                     # 右边界
                        if self.isoccupy([i, j+t], act1_matrix) == 1:
                            exit_flag = 1
                            break
                    if exit_flag == 1:      # 跳出
                        continue
                    for t in range(0, temp_l+1):                     # 下边界
                        if i-t<0:                   
                            # exit_flag = 1
                            break  
                        if self.isoccupy([i-t, j+temp_w], act1_matrix) == 1:
                            exit_flag = 1
                            break
                    if exit_flag == 1:      # 跳出
                        continue
                    for t in range(0, temp_w+1):                     # 左边界
                        if i-temp_l<0:       # 执行器左端的点不判断
                            break 
                        if self.isoccupy([i-temp_l, j+t], act1_matrix) == 1:
                            exit_flag = 1
                            break

                    if exit_flag == 0:   # 从第一个未被占用的像素点生成的矩阵符合插入条件
                        # p = temp
                        temp.x = i - (temp_l-1)/2 - 2   # 最后一个数为了留出GAP
                        temp.y = j + (temp_w-1)/2 + 2       
                        temp.r_cm = [temp.x, temp.y]
                        self.Parcels.append(copy.deepcopy(temp))         # 深度拷贝，避免修改后影响Parcels内的值

                        for m in range(i-temp_l,i+1):
                            for n in range(j,j+temp_w+1):
                                if m<0:
                                    m = 0
                                if n<0:
                                    n = 0
                                act1_matrix[m][n] = 1

                        prob = np.random.randint(1,11)   #1到10随机数
                        temp = actuator_array.Parcel(prob)

                        # insert_flag = 1
                        # break

            # if insert_flag == 1:
            #     break
        # end = time.time()
        # print(str(end-start))

    def isoccupy(self, pixel, act1_matrix):  # 判断像素点是否在当前像素矩阵中被占用
        # start = time.time()
        # for i in range(0,10000):
        try:
            if act1_matrix[pixel[0]][pixel[1]] == 1:
                return 1
        except:
            return 1  ##溢出直接不满足插入条件
        # if act1_matrix[pixel[0]][pixel[1]] == 1:
        #     return 1
        # end = time.time()

        # print(str(end-start))
        return 0


