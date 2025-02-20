#! /usr/bin/env python
import rospy
import math
import tf
import geometry_msgs.msg
import turtlesim.srv


def get_target_pos(target_turtle_pos, distance):
    """Get target position from target turtle position and distance.

    Args:
        target_turtle_pos (List): Target turtle position [x, y]
        distance (number): Distance to maintained
    """
    # 计算单位向量
    x, y = target_turtle_pos[0], target_turtle_pos[1]
    length = math.sqrt(x ** 2 + y ** 2)
    if length < 1e-2:
        return [0, 0]
    unit_x, unit_y = x / length, y / length

    # 计算要叠加的向量
    add_x = -unit_x * distance
    add_y = -unit_y * distance
    return [x + add_x, y + add_y]


def main():
    rospy.init_node('turtle_tf_listener')

    # 获取参数
    turtle_name = rospy.get_param('~turtle')
    target_turtle_name = rospy.get_param('~target_turtle')
    distance = rospy.get_param('~distance')

    # 调用 Service 生成海龟
    rospy.loginfo('waiting for service: spawn...')
    try:
        rospy.wait_for_service('spawn')
        spawner = rospy.ServiceProxy('spawn', turtlesim.srv.Spawn)
        spawner(4, 4, 0, turtle_name)
    except Exception as e:
        rospy.logerr(e)
        exit(-1)
    rospy.loginfo(f'spawn a turtle named: {turtle_name}')

    # 定义发布海龟速度的 Topic
    vel_pub = rospy.Publisher(f'{turtle_name}/cmd_vel', geometry_msgs.msg.Twist)

    # 定义 tf 监听器
    listener = tf.TransformListener()

    rate = rospy.Rate(10.0)
    rospy.loginfo(f'{turtle_name} is following {target_turtle_name}.')
    while not rospy.is_shutdown():
        try:
            # 获取 tf 变换
            listener.waitForTransform(f'/{turtle_name}', f'/{target_turtle_name}', rospy.Time(0), rospy.Duration(1))
            (trans, rot) = listener.lookupTransform(f'/{turtle_name}', f'/{target_turtle_name}', rospy.Time(0))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f'lookupTransform error: {e}')
            continue

        # 当前目标位置
        target_position = get_target_pos([trans[0], trans[1]], distance)

        # 防止距离目标点过近时，数值误差导致海龟原地转圈
        # 当海龟距离目标点很近时，不再进行控制（死区）
        if math.sqrt(target_position[0] ** 2 + target_position[1] ** 2) > 1e-2:
            # 用相对位置计算角速度、线速度（也可以用别的计算方法，合理即可）
            angular = 4 * math.atan2(target_position[1], target_position[0])
            linear = 0.5 * math.sqrt(target_position[0] ** 2 + target_position[1] ** 2)

            # 发布速度控制指令 Topic
            cmd = geometry_msgs.msg.Twist()
            cmd.linear.x = linear
            cmd.angular.z = angular
            vel_pub.publish(cmd)

        rate.sleep()


if __name__ == '__main__':
    main()
