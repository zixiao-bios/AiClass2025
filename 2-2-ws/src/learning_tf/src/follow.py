#! /usr/bin/env python
import rospy
import math
import tf
import geometry_msgs.msg

if __name__ == '__main__':
    rospy.init_node('turtle_tf_listener')

    turtle_name = rospy.get_param('~turtle_name')

    # 定义 tf 监听器
    listener = tf.TransformListener()

    # 定义发布海龟速度的 Topic
    turtle_vel = rospy.Publisher(f'{turtle_name}/cmd_vel', geometry_msgs.msg.Twist, queue_size=1)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            # 等待指定的 tf 变换可用
            listener.waitForTransform(turtle_name, '/turtle1', rospy.Time(0), rospy.Duration(1))

            # 查询 tf 树中，/turtle1 到 turtle_name 坐标系的变换（/turtle1 在 turtle_name 坐标系中的位姿）
            # 参数分别为：
            # 1. 目标坐标系名称
            # 2. 源坐标系名称
            # 3. 要查询的时刻，rospy.Time(0)表示查询最新的数据
            # 返回值：平移向量和四元数，即([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])
            (trans, rot) = listener.lookupTransform(turtle_name, '/turtle1', rospy.Time(0))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f'lookupTransform error: {e}')
            continue

        # 用相对位置计算角速度、线速度（也可以用别的计算方法，合理即可）
        
        # 用夹角大小确定角速度
        # math.atan2 计算从 x 轴正方向到点 (x, y) 的角度，单位为弧度，范围在 -pi 到 pi 之间
        angular = 4 * math.atan2(trans[1], trans[0])
        
        # 用距离确定线速度
        linear = 0.5 * math.sqrt(trans[0] ** 2 + trans[1] ** 2)

        # 发布速度控制指令 Topic
        cmd = geometry_msgs.msg.Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        turtle_vel.publish(cmd)

        rate.sleep()
