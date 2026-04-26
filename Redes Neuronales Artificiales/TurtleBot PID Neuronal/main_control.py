#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
import numpy as np
import math
import sys

from npid_directo import nPID_Directo
from npid_indirecto import nPID_Indirecto

class NeuralMasterNode(Node):
    def __init__(self):
        super().__init__('master_control')
        
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.sub = self.create_subscription(Odometry, '/odom', self.Callback, 10)
        
        # 'directo' o 'indirecto'
        self.tipo_control = 'directo' 

        self.d = 0.08
        self.last_time = 0.0
        
        self.eAx = 0.0; self.i_ex = 0.0
        self.eAy = 0.0; self.i_ey = 0.0
        
        self.u_x_last = 0.0; self.u_y_last = 0.0
        self.xp_last = 0.0;  self.yp_last = 0.0

        if self.tipo_control == 'directo':
            self.neurona_x = nPID_Directo(3, 0.5)
            self.neurona_y = nPID_Directo(3, 0.3)
        else:
            self.neurona_x = nPID_Indirecto(3, 0.5)
            self.neurona_y = nPID_Indirecto(3, 0.3)

    def Callback(self, msg_in):
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_time == 0.0:
            self.last_time = current_time
            self.xp_last = msg_in.pose.pose.position.x
            self.yp_last = msg_in.pose.pose.position.y
            return
        dt = current_time - self.last_time
        self.last_time = current_time
        
        x = msg_in.pose.pose.position.x
        y = msg_in.pose.pose.position.y
        q = msg_in.pose.pose.orientation
        theta = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        
        xd, yd = 3.0, 3.0
        xp = x + self.d * math.cos(theta)
        yp = y + self.d * math.sin(theta)
        
        ex = xd - xp
        ey = yd - yp
        
        self.i_ex += ex * dt
        self.i_ex = np.clip(self.i_ex, -1.0, 1.0)
        d_ex = (ex - self.eAx) / dt if dt > 0 else 0.0
        self.eAx = ex
        estado_x = np.array([ex, self.i_ex, d_ex])
        
        self.i_ey += ey * dt
        self.i_ey = np.clip(self.i_ey, -1.0, 1.0)
        d_ey = (ey - self.eAy) / dt if dt > 0 else 0.0
        self.eAy = ey
        estado_y = np.array([ey, self.i_ey, d_ey])
        
        if self.tipo_control == 'directo':
            kx = float(self.neurona_x.control_u(estado_x))
            self.neurona_x.fit(ex, estado_x)
            ky = float(self.neurona_y.control_u(estado_y))
            self.neurona_y.fit(ey, estado_y, 0.03)
            
        elif self.tipo_control == 'indirecto':
            kx = float(self.neurona_x.control_u(estado_x))
            self.neurona_x.fit(error_trayectoria=ex, x=estado_x, u_aplicado=self.u_x_last, salida_real=xp)
            ky = float(self.neurona_y.control_u(estado_y))
            self.neurona_y.fit(error_trayectoria=ey, x=estado_y, u_aplicado=self.u_y_last, salida_real=yp)

        matriz_modelo = np.array([[np.cos(theta), -self.d*np.sin(theta)],
                                  [np.sin(theta),  self.d*np.cos(theta)]])
        
        u = np.dot(np.linalg.inv(matriz_modelo), np.array([[kx], [ky]]))
        
        self.u_x_last = kx
        self.u_y_last = ky
        self.xp_last = xp
        self.yp_last = yp
        
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = np.clip(float(u[0][0]), -0.26, 0.26)
        msg.twist.angular.z = np.clip(float(u[1][0]), -1.82, 1.82)
        
        if abs(ex) < 0.01 and abs(ey) < 0.01:
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.0
            self.pub.publish(msg)
            print(f"¡Llegó el Robot {self.tipo_control.upper()}!")
            sys.exit()
            
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    nodo = NeuralMasterNode()
    try:
        rclpy.spin(nodo)
    except SystemExit:
        pass
    nodo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
