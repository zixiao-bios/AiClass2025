#! /usr/bin/env python
import rospy
import random
from geometry_msgs.msg import Twist

if __name__ == '__main__':
    rospy.init_node("turtle1_control")

    turtle_name = rospy.get_param("~turtle_name")

    # 创建速度控制 Topic 发布者
    pub = rospy.Publisher(f'{turtle_name}/cmd_vel', Twist)

    # 发布的消息变量
    control_msg = Twist()
    control_msg.linear.x = 5 * random.random()
    control_msg.angular.z = random.choice([1, -1]) * random.uniform(0.5, 5)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rospy.loginfo(control_msg)
        pub.publish(control_msg)
        rate.sleep()
