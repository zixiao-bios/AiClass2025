#! /usr/bin/env python
import rospy
import turtlesim.srv

if __name__ == '__main__':
    rospy.init_node('spawn_turtle')

    turtle_name = rospy.get_param("turtle_name")

    # 等待 service 存在，参数：服务名
    rospy.wait_for_service('spawn')

    # 生成 ServiceProxy（把服务转为一个可调用的函数）
    # 参数：服务名，服务消息类型
    spawner = rospy.ServiceProxy('spawn', turtlesim.srv.Spawn)

    # 调用服务，参数：服务的参数
    spawner(4, 2, 0, turtle_name)
