#! /usr/bin/env python
import rospy
from std_msgs.msg import String

def talker():
    # 向 ROS Master 注册名为 talker 的节点
    rospy.init_node('talker')

    # 创建 Topic 发布者，Topic 名称为 test_msg，数据类型为 String
    pub = rospy.Publisher('test_msg', String)

    # 定义 2Hz 的 rate 变量，即每次 sleep 时长为 0.5s
    rate = rospy.Rate(2)

    # 在节点没有关闭时，执行循环
    while not rospy.is_shutdown():
        msg = f'Hello ROS! From talker, at {rospy.get_time()}'

        # 在终端打印日志
        rospy.loginfo(msg)

        # 发布 Topic
        pub.publish(msg)

        # 等待 0.5s
        rate.sleep()
    
    rospy.loginfo('Talker exist.')

if __name__ == '__main__':
    talker()
