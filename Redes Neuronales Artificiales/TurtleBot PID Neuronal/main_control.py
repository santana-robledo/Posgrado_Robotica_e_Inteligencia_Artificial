#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped

from npid_directo import nPID_Directo
from npid_indirecto import nPID_Indirecto


class NeuralMasterNode(Node):

    def __init__(self):
        super().__init__('master_control')

        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.sub = self.create_subscription(Odometry, '/odom', self.callback, 10)

        self.tipo_control = 'directo'

        # Punto adelantado
        self.d = 0.12

        # Tiempo
        self.last_time = None
        self.last_print_time = None
        self.shutdown_started = False

        # Errores previos
        self.eAx = None
        self.eAy = None

        # Integrales
        self.i_ex = 0.0
        self.i_ey = 0.0

        # Derivadas filtradas
        self.d_ex_f = 0.0
        self.d_ey_f = 0.0
        self.deriv_filter = 0.9

        # Último control
        self.u_x_last = 0.0
        self.u_y_last = 0.0

        # Referencia
        self.xd = 1.0
        self.yd = 1.0

        # Estado
        self.goal_reached = False

        # Cambio a control final
        self.switch_to_p = 0.35

        # Ganancia final más fuerte
        self.kp_final = 1.8

        # Redes neuronales
        if self.tipo_control == 'directo':
            self.neurona_x = nPID_Directo(3, beta=0.3)
            self.neurona_y = nPID_Directo(3, beta=0.3)
        else:
            self.neurona_x = nPID_Indirecto(4, beta=0.3)
            self.neurona_y = nPID_Indirecto(4, beta=0.3)

    def callback(self, msg_in):
        if self.goal_reached:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.last_time is None:
            self.last_time = current_time
            self.last_print_time = current_time
            return

        dt = max(current_time - self.last_time, 1e-3)
        self.last_time = current_time

        # =========================
        # Pose actual
        # =========================
        x = msg_in.pose.pose.position.x
        y = msg_in.pose.pose.position.y

        q = msg_in.pose.pose.orientation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # =========================
        # Punto adelantado
        # =========================
        xp = x + self.d * math.cos(theta)
        yp = y + self.d * math.sin(theta)

        # Error del punto adelantado
        ex = self.xd - xp
        ey = self.yd - yp
        distancia_error = math.hypot(ex, ey)

        # Error real del robot
        ex_real = self.xd - x
        ey_real = self.yd - y
        distancia_real = math.hypot(ex_real, ey_real)

        # =========================
        # Llegada real
        # =========================
        if distancia_real < 0.01:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.0
            self.pub.publish(msg)

            self.goal_reached = True
            print(f"x={x:>+6.3f}, y={y:>+6.3f}")
            print("\nMeta alcanzada")

            if not self.shutdown_started:
                self.shutdown_started = True
                self.create_timer(0.5, self.shutdown_cleanly)
            return

        # =========================
        # Integral sobre punto adelantado
        # =========================
        self.i_ex += ex * dt
        self.i_ey += ey * dt

        self.i_ex = np.clip(self.i_ex, -5.0, 5.0)
        self.i_ey = np.clip(self.i_ey, -5.0, 5.0)

        if distancia_error < 0.04:
            self.i_ex *= 0.99
            self.i_ey *= 0.99

        # =========================
        # Derivadas
        # =========================
        if self.eAx is None:
            d_ex = 0.0
            d_ey = 0.0
        else:
            d_ex = (ex - self.eAx) / dt
            d_ey = (ey - self.eAy) / dt

        self.eAx = ex
        self.eAy = ey

        self.d_ex_f = self.deriv_filter * self.d_ex_f + (1.0 - self.deriv_filter) * d_ex
        self.d_ey_f = self.deriv_filter * self.d_ey_f + (1.0 - self.deriv_filter) * d_ey

        # =========================
        # Normalización
        # =========================
        ex_n = np.clip(ex, -5.0, 5.0) / 5.0
        ey_n = np.clip(ey, -5.0, 5.0) / 5.0
        iex_n = np.clip(self.i_ex, -5.0, 5.0) / 5.0
        iey_n = np.clip(self.i_ey, -5.0, 5.0) / 5.0
        dex_n = np.clip(self.d_ex_f, -10.0, 10.0) / 10.0
        dey_n = np.clip(self.d_ey_f, -10.0, 10.0) / 10.0

        # =========================
        # Control híbrido
        # =========================
        if distancia_real > self.switch_to_p:
            # Control neuronal lejos
            if self.tipo_control == 'directo':
                estado_x = np.array([ex_n, iex_n, dex_n], dtype=float)
                estado_y = np.array([ey_n, iey_n, dey_n], dtype=float)

                kx = float(self.neurona_x.control_u(estado_x))
                ky = float(self.neurona_y.control_u(estado_y))

                if distancia_error > 0.12:
                    self.neurona_x.fit(ex_n, estado_x)
                    self.neurona_y.fit(ey_n, estado_y)
            else:
                estado_x_id = np.array([ex_n, iex_n, dex_n, self.u_x_last], dtype=float)
                estado_y_id = np.array([ey_n, iey_n, dey_n, self.u_y_last], dtype=float)

                kx = float(self.neurona_x.control_u(estado_x_id))
                ky = float(self.neurona_y.control_u(estado_y_id))

                if distancia_error > 0.12:
                    self.neurona_x.fit(ex_n, estado_x_id)
                    self.neurona_y.fit(ey_n, estado_y_id)
        else:
            # Control proporcional final usando error REAL
            kx = self.kp_final * ex_real
            ky = self.kp_final * ey_real

        self.u_x_last = kx
        self.u_y_last = ky

        # =========================
        # Cinemática inversa
        # =========================
        matriz_modelo = np.array([
            [math.cos(theta), -self.d * math.sin(theta)],
            [math.sin(theta),  self.d * math.cos(theta)]
        ], dtype=float)

        try:
            u = np.linalg.solve(matriz_modelo, np.array([kx, ky], dtype=float))
        except np.linalg.LinAlgError:
            u = np.array([0.0, 0.0], dtype=float)

        v = float(u[0])
        w = float(u[1])

        # =========================
        # Suavizado final más débil
        # =========================
        if distancia_real < 0.20:
            factor_v = max(distancia_real / 0.20, 0.65)
            factor_w = max(distancia_real / 0.20, 0.55)
            v *= factor_v
            w *= factor_w

        # Saturaciones
        v = np.clip(v, -0.26, 0.26)
        w = np.clip(w, -1.82, 1.82)

        # =========================
        # Publicación
        # =========================
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.pub.publish(msg)

        # =========================
        # Impresión
        # =========================
        if current_time - self.last_print_time >= 0.5:
            print(f"x={x:>+6.3f}, y={y:>+6.3f}")
            self.last_print_time = current_time

    def shutdown_cleanly(self):
        if self.goal_reached and rclpy.ok():
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    nodo = NeuralMasterNode()

    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            nodo.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()