// Stonefish 无界面仿真核心：加载场景、按控制步推进、在 btMultiBody 层面精确读写状态。
//
// 每个控制步调用 decimation 次 StepSimulation(fixed_dt)。fixed_dt 与 Stonefish 内部的 ssus/1e6
// 完全相同，保证每次调用恰好推进一个物理子步，不让 Bullet 的时间累加器抖动。
//
// 两种状态表示：
//   State（原生）：基座 = btMultiBody 基座（复合体的质心 + 惯量主轴坐标系），精确读写，供基准测试用。
//   body 向量   ：基座 = 场景里 base_link 的原点坐标系（x 向前、z 向下），速度为该原点的速度。
//                  [pos(3), quat wxyz(4), lin_vel_world(3), ang_vel_world(3), q(nq), qd(nq)]
// 两者之间只差一个常量刚体变换（base_cg2o），换算是精确的（浮点舍入级）。
//
// Stonefish 的 SimulationApp 是单例，所以一个进程只能有一个 Sim。

#pragma once

#include <core/ConsoleSimulationApp.h>
#include <core/FeatherstoneRobot.h>
#include <core/ScenarioParser.h>
#include <core/SimulationManager.h>
#include <entities/FeatherstoneEntity.h>
#include <entities/SolidEntity.h>
#include <actuators/Servo.h>
#include <actuators/Thruster.h>
#include <BulletDynamics/Featherstone/btMultiBody.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace sfsim {

class Manager : public sf::SimulationManager {
 public:
  Manager(sf::Scalar sps, std::string scn) : sf::SimulationManager(sps), scn_(std::move(scn)) {}
  void BuildScenario() override {
    sf::ScenarioParser parser(this);
    if (!parser.Parse(scn_)) throw std::runtime_error("scenario parsing failed: " + scn_);
  }

 private:
  std::string scn_;
};

// 暴露 ConsoleSimulationApp 的受保护 Init()，不启动它自带的实时仿真线程。
class App : public sf::ConsoleSimulationApp {
 public:
  using sf::ConsoleSimulationApp::ConsoleSimulationApp;
  void Setup() { Init(); }
};

struct State {
  btTransform base;  // btMultiBody 基座（质心系）
  btVector3 v, w;    // 世界系线速度/角速度
  std::vector<double> q, qd;
};

struct Sim {
  Manager* sm = nullptr;
  App* app = nullptr;
  sf::FeatherstoneRobot* robot = nullptr;
  sf::FeatherstoneEntity* dyn = nullptr;
  btMultiBody* mb = nullptr;
  std::vector<sf::Thruster*> thrusters;
  std::vector<sf::Servo*> servos;
  std::vector<int> dof_links;          // 有自由度（转动/移动）的 btMultiBody link 下标
  std::vector<std::string> dof_names;  // 与 dof_links 一一对应的关节名
  btTransform base_cg2o;               // 基座质心系 -> base_link 原点系
  double fixed_dt = 0.0;
  int decimation = 1;
  int fd_prescaler = 1;

  void init(const std::string& scn, const std::string& data, double phys_hz, double ctrl_hz) {
    sm = new Manager(phys_hz, scn);
    app = new App("sfsim", data, sm);
    app->Setup();
    if (!sm->StartSimulation()) throw std::runtime_error("initial-condition solve failed");
    fixed_dt = double(uint64_t(1000000.0 / phys_hz)) / 1000000.0;
    decimation = int(std::lround((1.0 / ctrl_hz) / fixed_dt));
    if (decimation < 1) throw std::runtime_error("ctrl_hz must not exceed phys_hz");
    // 流体力每 round(sps/50) 个子步才重算一次（SimulationManager 内部的计数器是全局的）。
    // decimation 是它的整数倍时，每个控制步内重算发生在相同相位，状态重置后结果才可复现。
    fd_prescaler = std::max(1, int(std::lround(phys_hz / 50.0)));
    if (decimation % fd_prescaler != 0)
      std::fprintf(stderr, "[sfsim] warning: decimation %d is not a multiple of the fluid-dynamics prescaler %d\n",
                   decimation, fd_prescaler);
    bind();
  }

  void bind() {
    robot = dynamic_cast<sf::FeatherstoneRobot*>(sm->getRobot(0));
    if (robot == nullptr) throw std::runtime_error("robot 0 is not a FeatherstoneRobot");
    dyn = robot->getDynamics();
    mb = dyn->getMultiBody();
    thrusters.clear();
    servos.clear();
    for (size_t i = 0;; ++i) {
      sf::Actuator* a = robot->getActuator(i);
      if (a == nullptr) break;
      if (auto* t = dynamic_cast<sf::Thruster*>(a)) thrusters.push_back(t);
      if (auto* s = dynamic_cast<sf::Servo*>(a)) servos.push_back(s);
    }
    dof_links.clear();
    for (int i = 0; i < mb->getNumLinks(); ++i)
      if (mb->getLink(i).m_dofCount == 1) dof_links.push_back(i);
    // FeatherstoneEntity 的关节 j 连接 parent -> child（Stonefish link 下标，0 为基座），
    // 对应 btMultiBody 的 link 下标为 child - 1。
    dof_names.assign(dof_links.size(), "");
    for (unsigned j = 0; j < dyn->getNumOfJoints(); ++j) {
      sf::FeatherstoneJoint jt = dyn->getJoint(j);
      for (size_t k = 0; k < dof_links.size(); ++k)
        if (int(jt.child) - 1 == dof_links[k]) dof_names[k] = jt.name;
    }
    base_cg2o = dyn->getLink(0).solid->getCG2OTransform();
  }

  int nt() const { return int(thrusters.size()); }
  int ns() const { return int(servos.size()); }
  int nq() const { return int(dof_links.size()); }
  int na() const { return nt() + ns(); }
  int nx() const { return 13 + 2 * nq(); }
  int action_dim() const { return na(); }
  double ctrl_dt() const { return fixed_dt * decimation; }

  void apply(const double* a) {
    for (int i = 0; i < nt(); ++i) thrusters[i]->setSetpoint(a[i]);
    for (int j = 0; j < ns(); ++j) servos[j]->setDesiredPosition(a[nt() + j]);
  }
  void apply(const std::vector<double>& a) { apply(a.data()); }

  void step_ctrl() {
    for (int k = 0; k < decimation; ++k) sm->StepSimulation(fixed_dt);
  }

  State get() const {
    State s;
    s.base = mb->getBaseWorldTransform();
    s.v = mb->getBaseVel();
    s.w = mb->getBaseOmega();
    for (int i : dof_links) {
      s.q.push_back(mb->getJointPos(i));
      s.qd.push_back(mb->getJointVel(i));
    }
    return s;
  }

  void set(const State& s) {
    mb->setBaseWorldTransform(s.base);
    mb->setBaseVel(s.v);
    mb->setBaseOmega(s.w);
    for (size_t k = 0; k < dof_links.size(); ++k) {
      mb->setJointPos(dof_links[k], s.q[k]);
      mb->setJointVel(dof_links[k], s.qd[k]);
    }
    mb->clearForcesAndTorques();
    mb->clearConstraintForces();
    btAlignedObjectArray<btQuaternion> sq;
    btAlignedObjectArray<btVector3> sv;
    mb->forwardKinematics(sq, sv);
    mb->updateCollisionObjectWorldTransforms(sq, sv);
  }

  // ---- body 向量（base_link 原点系）----
  void get_body(double* x) const {
    const btTransform cg = mb->getBaseWorldTransform();
    const btTransform o = cg * base_cg2o;
    const btVector3 w = mb->getBaseOmega();
    const btVector3 v = mb->getBaseVel() + w.cross(o.getOrigin() - cg.getOrigin());
    const btQuaternion r = o.getRotation();
    const btVector3& p = o.getOrigin();
    x[0] = p.x(); x[1] = p.y(); x[2] = p.z();
    x[3] = r.getW(); x[4] = r.getX(); x[5] = r.getY(); x[6] = r.getZ();
    x[7] = v.x(); x[8] = v.y(); x[9] = v.z();
    x[10] = w.x(); x[11] = w.y(); x[12] = w.z();
    for (int k = 0; k < nq(); ++k) {
      x[13 + k] = mb->getJointPos(dof_links[k]);
      x[13 + nq() + k] = mb->getJointVel(dof_links[k]);
    }
  }

  void set_body(const double* x) {
    btTransform o(btQuaternion(x[4], x[5], x[6], x[3]).normalized(), btVector3(x[0], x[1], x[2]));
    State s;
    s.base = o * base_cg2o.inverse();
    s.w = btVector3(x[10], x[11], x[12]);
    s.v = btVector3(x[7], x[8], x[9]) + s.w.cross(s.base.getOrigin() - o.getOrigin());
    s.q.assign(x + 13, x + 13 + nq());
    s.qd.assign(x + 13 + nq(), x + 13 + 2 * nq());
    set(s);
  }

  // 所有 link 的原点系世界位姿，(nlinks, 7)：pos(3) + quat wxyz(4)。link 0 为基座。
  int nlinks() const { return int(dyn->getNumOfLinks()); }
  std::string link_name(int i) const { return dyn->getLink(i).solid->getName(); }
  void link_poses(double* out) const {
    for (int i = 0; i < nlinks(); ++i) {
      const btTransform t = i == 0 ? mb->getBaseWorldTransform() * base_cg2o : dyn->getLink(i).solid->getOTransform();
      const btQuaternion r = t.getRotation();
      double* o = out + 7 * i;
      o[0] = t.getOrigin().x(); o[1] = t.getOrigin().y(); o[2] = t.getOrigin().z();
      o[3] = r.getW(); o[4] = r.getX(); o[5] = r.getY(); o[6] = r.getZ();
    }
  }
};

}  // namespace sfsim
