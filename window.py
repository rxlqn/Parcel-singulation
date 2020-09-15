# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Actuator_array.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import QRect, QPoint, QSize, QTimer, QRectF
from PyQt5.QtWidgets import QWidget, QLabel

from math import pi, cos, sin, floor
import sys
import matplotlib.pyplot as plt
import numpy as np

import time
import copy
import simulation as si
import control_policy as ctl

import multiprocessing as mp
import tensorflow as tf
import AC_CartPole as ac 

LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

# env = gym.make('CartPole-v0')
# env.seed(1)  # reproducible
# env = env.unwrapped   

# N_F = env.observation_space.shape[0]
# N_A = env.action_space.n
N_F = 8         # # of features
N_A = 5*17        # # of actions      17个传送带 

env = si.Environment()
s = env.reset()

class Ui_MainWindow(QWidget):       # 继承Qwidget
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.but1 = QtWidgets.QPushButton(self.centralwidget)
        self.but1.setGeometry(QtCore.QRect(580, 510, 88, 23))
        self.but1.setObjectName("but1")
        self.combo1 = QtWidgets.QComboBox(self.centralwidget)
        self.combo1.setGeometry(QtCore.QRect(670, 250, 89, 21))
        self.combo1.setObjectName("combo1")
        self.combo1.addItem("")
        self.combo2 = QtWidgets.QComboBox(self.centralwidget)
        self.combo2.setGeometry(QtCore.QRect(670, 380, 89, 21))
        self.combo2.setObjectName("combo2")
        self.combo2.addItem("")

        # # #定义要绘制图像的结构
        # self.paint_array = Actuator_array()

        self.lb = mylabel(self.centralwidget)    # 重写的Qlabel,自动触发paintevent
        self.lb.setGeometry(QtCore.QRect(40, 30, 590, 440))

        # 绘制table
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(40, 300, 500, 300))
        self.tableWidget.setObjectName("tableWidget")

        self.tableWidget.setColumnCount(act_array.col)
        self.tableWidget.setRowCount(act_array.row)

        for i in range(0, act_array.row):
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setVerticalHeaderItem(i, item)

        for i in range(0, act_array.col):
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setHorizontalHeaderItem(i, item)

        for i in range(0, act_array.row):
            for j in range(0, act_array.col):
                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i, j, item)

        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(680, 220, 72, 15))
        self.label1.setObjectName("label1")
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(690, 350, 72, 15))
        self.label2.setObjectName("label2")
        self.but2 = QtWidgets.QPushButton(self.centralwidget)
        self.but2.setGeometry(QtCore.QRect(680, 510, 88, 23))
        self.but2.setObjectName("but2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # 添加combo中的数字
        for i in range(1, act_array.row*act_array.col+1):
            self.combo1.addItem(str(i))
        for i in range(1, 101):
            self.combo2.addItem(str(i))

        self.but1.clicked.connect(self.pressed)
        self.but2.clicked.connect(self.reset)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.refresh()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.but1.setText(_translate("MainWindow", "Start"))
        self.combo1.setItemText(0, _translate("MainWindow", "0"))
        self.combo2.setItemText(0, _translate("MainWindow", "0"))

        self.label1.setText(_translate("MainWindow", "Actuator"))
        self.label2.setText(_translate("MainWindow", "Speed"))
        self.but2.setText(_translate("MainWindow", "Reset"))

        for i in range(0, act_array.row):
            item = self.tableWidget.verticalHeaderItem(i)
            item.setText(_translate("MainWindow", str(i)))

        for i in range(0, act_array.col):
            item = self.tableWidget.horizontalHeaderItem(i)
            item.setText(_translate("MainWindow", str(i)))

        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)

        item = self.tableWidget.item(0, 0)
        item.setText(_translate("MainWindow", "test"))

        self.tableWidget.setSortingEnabled(__sortingEnabled)

        # 更新table
        for i in range(0, act_array.row):
            for j in range(0, act_array.col):
                index = i*act_array.col+j+1
                table_speed = act_array.actuator[index].speed
                item = self.tableWidget.item(i, j)
                item.setText(str(table_speed))

    def refresh(self):
        fps = 10
        self.timer = QTimer(self)
        self.timer.setInterval(1000/fps)
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start()
        self.timer_1s = QTimer(self)
        self.timer_1s.setInterval(1000)
        self.timer_1s.timeout.connect(self.timer1s_Event)
        self.timer_1s.start()

    def get_tablespeed(self):
        global act_array
        for i in range(0, act_array.row):
            for j in range(0, act_array.col):
                index = i*act_array.col+j+1
                # speed = int(self.tableWidget.item(i, j).text())
                speed = act_array.actuator[index].speed
                (red, green, blue) = self.num2rgb(speed)
                act_array.actuator[index].color = QColor(red, green, blue)

                item = self.tableWidget.item(i, j)
                item.setText(str(speed))
        # 0传送带
        speed = act_array.actuator[0].speed
        (red, green, blue) = self.num2rgb(speed)
        act_array.actuator[0].color = QColor(red, green, blue)

    def pressed(self):  # 按键中断
        act_index = int(self.combo1.currentText())
        speed = int(self.combo2.currentText())
        global act_array
        act_array.actuator[act_index].speed = speed

        # 按照配置的speed热力图显示
        (red, green, blue) = self.num2rgb(speed)
        act_array.actuator[act_index].color = QColor(red, green, blue)

    def reset(self):
        print("reset")
        # for i in range(0, 3):        # 随机生成两个包裹
        #     Parcels.append(actuator_array.Parcel())
        # Parcels[0].y = np.random.randint(25, 55)
        # Parcels[0].r_cm[1] = Parcels[0].y
        # Parcels[1].y = np.random.randint(60, 100)
        # Parcels[1].r_cm[1] = Parcels[1].y
        # Parcels[2].y = np.random.randint(120, 220)
        # Parcels[2].r_cm[1] = Parcels[2].y
        # Parcels[3].y = np.random.randint(120,220)
        # Parcels[3].r_cm[1] = Parcels[1].y

    def num2rgb(self, num):

        color = [[0, 0, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
        normal = float(num)/101
        normal = normal*3
        idx1 = int(normal)
        idx2 = int(idx1 + 1)
        fractBetween = normal - idx1
        red = ((color[idx2][0] - color[idx1][0])*fractBetween + color[idx1][0])
        green = ((color[idx2][1] - color[idx1][1])
                 * fractBetween + color[idx1][1])
        blue = ((color[idx2][2] - color[idx1][2])
                * fractBetween + color[idx1][2])
        return red, green, blue


    def timer1s_Event(self):        # 生成包裹
        # simulation.Generate_parcels()
        pass


    def timerEvent(self):                       # 10fps 刷新中断，可以执行仿真

        global act_array 
        global Parcels
        global s
        # # 仿真运行
        # start = time.time()

        # simulation.Parcel_sim()

        # end = time.time()
        # print("仿真时间:",str(end-start))

        # act_array = simulation.act_array   
        # Parcels = simulation.Parcels      # 返回当前包裹的信息
        Parcels = env.Parcels
        act_array = env.act_array
        # 控制策略
        start = time.time()

        # if len(Parcels)!=0:
        #     ctl.Control(simulation)

        a = actor.choose_action(s)
        
        s_, r, done = env.step(a)
        s = s_

        end = time.time()
        # print("控制策略:",str(end-start))
        # print()
        # 图形化更新
        self.get_tablespeed()             
 
        self.lb.update()                        # 渲染更新


class mylabel(QLabel):              # 继承 Qlabel 在label范围内画图
    def paintEvent(self, event):
        global act_array
        super().paintEvent(event)
        painter = QPainter(self)
        for i in range(0, act_array.row*act_array.col+1):
            painter.drawRect(act_array.rec[i])
            bs = QBrush(act_array.actuator[i].color)
            painter.fillRect(act_array.rec[i], bs)

        # # 含角度显示
        for index in range(0, len(Parcels)):

            painter.resetTransform()        # 重置变换
            parcel = Parcels[index]
            rec_x = parcel.x - (parcel.l-1)/2
            rec_y = parcel.y - (parcel.w-1)/2
            parcel.rec = QRectF(rec_x, rec_y, parcel.l, parcel.w)

            # theta = -parcel.theta/180*pi   # 先反方向旋转
            # # 坐标y轴是反的，所以想要使用旋转矩阵需要反过来乘
            # M = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
            # [rec_x2, rec_y2] = np.dot([parcel.x, parcel.y], M)
            # painter.rotate(-parcel.theta)  # 坐标旋转角度
            # parcel.rec = parcel.rec.translated(
            #     int(rec_x2-parcel.x), int(rec_y2-parcel.y))
            painter.drawRect(parcel.rec)
            bs = QBrush(QColor(255, 255, 255))
            painter.fillRect(parcel.rec, bs)

            painter.drawText(parcel.rec,str(parcel.prio))
        # 绘制测试代码
        # painter.setPen(QColor(0,0,0))
        # painter.drawLine(300,300,205,87)
        # print(act_array.cal_num_act(205,87))


act_array = 0  # 全局变量
Parcels = 0
simualation = 0

if __name__ == "__main__":


    
    sess = tf.Session()

    actor = ac.Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = ac.Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor


    saver = tf.train.Saver()
    ckpt_path = './ckpt/test-model.ckpt'
    saver.restore(sess, ckpt_path)

    # s = env.reset()

    simulation = si.Physic_simulation()
    act_array = simulation.act_array  # 全局变量
    Parcels = simulation.Parcels
    simulation.Generate_parcels()

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    #####


    ## 一个进程跑仿真和显示
    ## 一个进程跑控制策略

    ####
    sys.exit(app.exec_())
