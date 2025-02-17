#! /usr/bin/env python
import rospy
from std_msgs.msg import String

def get_test_msg(data):
    # data.data 中是 Topic 数据
    rospy.loginfo(f'Listener get msg: {data.data}')

def listener():
    rospy.init_node('listener')

    # 创建 Topic 订阅者，订阅名为 test_msg 的 Topic，数据类型为 String，收到 Topic 后的回调函数是 get_test_msg
    rospy.Subscriber('test_msg', String, get_test_msg)

    # 防止程序提前退出，等到节点被关闭时再退出
    rospy.spin()

if __name__ == '__main__':
    listener()
