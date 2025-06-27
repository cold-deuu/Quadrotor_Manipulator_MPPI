#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Header.h>
#include <iostream>
#include <string>
#include <vector>

namespace gazebo
{
class JointStateSensor : public ModelPlugin
{
public:
  ros::NodeHandle nh;
  std::string namespc;
  ros::Publisher joint_state_pub;
  uint32_t sequence = 0;
  physics::ModelPtr model;
  event::ConnectionPtr updateConnection;

  void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf) override
  {
    this->model = _parent;
    this->model->SetSelfCollide(false);

    if (_sdf->HasElement("nameSpace"))
      namespc = _sdf->Get<std::string>("nameSpace");
    else
      namespc = "default";

    std::string topicName = namespc + "/joint_info";
    joint_state_pub = nh.advertise<sensor_msgs::JointState>(topicName, 1);

    this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&JointStateSensor::onUpdate, this));
  }

  void onUpdate()
  {
    sequence++;
    ros::Time curr_time = ros::Time::now();

    sensor_msgs::JointState joint_msg;
    joint_msg.header.seq = sequence;
    joint_msg.header.stamp = curr_time;
    joint_msg.header.frame_id = "";

    std::vector<std::string> names;
    std::vector<double> angles, velocities, efforts;

    std::string standard_prefix = namespc + "::" + namespc + "/land";

    for (int i = 1; i <= 2; ++i)
    {
      std::string joint_name = standard_prefix + std::to_string(i) + "_joint";
      physics::JointPtr joint = this->model->GetJoint(joint_name);

      if (!joint)
      {
        gzerr << "[JointStateSensor] Joint not found: " << joint_name << "\n";
        continue;
      }

      double speed = joint->GetVelocity(0);
      double angle = joint->Position(0); // ✅ Gazebo 11 방식

      names.emplace_back(joint_name);
      angles.emplace_back(angle);
      velocities.emplace_back(speed);
      efforts.emplace_back(0.0);  // effort 미지원 시 0으로
    }

    joint_msg.name = names;
    joint_msg.position = angles;
    joint_msg.velocity = velocities;
    joint_msg.effort = efforts;

    joint_state_pub.publish(joint_msg);
  }
};

GZ_REGISTER_MODEL_PLUGIN(JointStateSensor)
}  // namespace gazebo
