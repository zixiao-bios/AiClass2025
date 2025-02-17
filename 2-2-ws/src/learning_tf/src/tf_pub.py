#! /usr/bin/env python  
import rospy
import tf
from turtlesim.msg import Pose


def handle_pose(msg, turtle_name):
    br = tf.TransformBroadcaster()

    p = (msg.x, msg.y, 0)

    br.sendTransform(
        p,
        tf.transformations.quaternion_from_euler(0, 0, msg.theta),
        rospy.Time.now(),
        turtle_name,
        'world'
    )


if __name__ == '__main__':
    rospy.init_node('tf_pub')

    turtle_name = rospy.get_param('~turtle_name')

    rospy.Subscriber(f'/{turtle_name}/pose', Pose, handle_pose, turtle_name)

    rospy.spin()
