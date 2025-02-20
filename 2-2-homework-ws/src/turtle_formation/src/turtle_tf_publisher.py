#! /usr/bin/env python  
import rospy
import tf
from turtlesim.msg import Pose

def handle_turtle_pose(msg, turtlename):
    # 定义一个 tf 广播器
    br = tf.TransformBroadcaster()

    # 发送一个 tf 变换
    br.sendTransform((msg.x, msg.y, 0),
                     tf.transformations.quaternion_from_euler(0, 0, msg.theta),
                     rospy.Time.now(),
                     turtlename,
                     "world")

if __name__ == '__main__':
    rospy.init_node('turtle_tf_broadcaster')

    # 从参数中获取要发布 tf 的海龟名称，"~"开头表示私有参数，即在launch文件node标签内部的参数
    turtlename = rospy.get_param('~turtle')

    # 订阅海龟位置的 Topic，最后一个参数是额外传入回调函数的参数，相应的回调函数的参数变成了两个
    rospy.Subscriber(f'/{turtlename}/pose', Pose, handle_turtle_pose, turtlename)

    rospy.spin()
