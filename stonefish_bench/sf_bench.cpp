// Stonefish 无界面基准：吞吐 / 状态重置一致性 / 运行时域随机化 / 重建开销。
//
// 用法:
//   sf_bench throughput <scenario.scn> <data_dir> [phys_hz=500] [ctrl_hz=50] [sim_seconds=120] [episode_steps=500] [seed=0]
//   sf_bench reset      <scenario.scn> <data_dir> [phys_hz=500] [ctrl_hz=50]
//   sf_bench current    <scenario.scn> <data_dir> [phys_hz=500] [ctrl_hz=50]
//   sf_bench rebuild    <scenario.scn> <data_dir> [phys_hz=500]
//
// 每个控制步调用 decimation 次 StepSimulation(fixed_dt)，fixed_dt 与 Stonefish 内部的
// ssus/1e6 完全相同，保证每次调用恰好推进一个物理子步（不让 Bullet 的时间累加器抖动）。
// 机器人状态直接在 btMultiBody 层面读写（基座位姿/速度 + 各关节 q/qd），不经过任何传感器。

#include <core/ConsoleSimulationApp.h>
#include <core/FeatherstoneRobot.h>
#include <core/ScenarioParser.h>
#include <core/SimulationManager.h>
#include <entities/FeatherstoneEntity.h>
#include <entities/forcefields/Ocean.h>
#include <entities/forcefields/Uniform.h>
#include <actuators/Servo.h>
#include <actuators/Thruster.h>
#include <BulletDynamics/Featherstone/btMultiBody.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

class BenchManager : public sf::SimulationManager {
 public:
  BenchManager(sf::Scalar sps, std::string scn) : sf::SimulationManager(sps), scn_(std::move(scn)) {}
  void BuildScenario() override {
    sf::ScenarioParser parser(this);
    if (!parser.Parse(scn_)) {
      std::fprintf(stderr, "[bench] scenario parsing failed: %s\n", scn_.c_str());
      std::exit(2);
    }
  }

 private:
  std::string scn_;
};

// 暴露 ConsoleSimulationApp 的受保护 Init()，不启动它自带的实时仿真线程。
class BenchApp : public sf::ConsoleSimulationApp {
 public:
  using sf::ConsoleSimulationApp::ConsoleSimulationApp;
  void Setup() { Init(); }
};

struct State {
  btTransform base;
  btVector3 v, w;  // 世界系基座线速度/角速度（btMultiBody 原生表示）
  std::vector<double> q, qd;
};

struct Sim {
  BenchManager* sm = nullptr;
  BenchApp* app = nullptr;
  sf::FeatherstoneRobot* robot = nullptr;
  btMultiBody* mb = nullptr;
  std::vector<sf::Thruster*> thrusters;
  std::vector<sf::Servo*> servos;
  std::vector<int> dof_links;  // 有自由度（转动/移动）的 link 下标
  double fixed_dt = 0.0;
  int decimation = 1;

  void init(const std::string& scn, const std::string& data, double phys_hz, double ctrl_hz) {
    sm = new BenchManager(phys_hz, scn);
    app = new BenchApp("sf_bench", data, sm);
    app->Setup();
    if (!sm->StartSimulation()) {
      std::fprintf(stderr, "[bench] initial-condition solve failed\n");
      std::exit(3);
    }
    fixed_dt = double(uint64_t(1000000.0 / phys_hz)) / 1000000.0;
    decimation = int(std::lround((1.0 / ctrl_hz) / fixed_dt));
    bind();
  }

  void bind() {
    robot = dynamic_cast<sf::FeatherstoneRobot*>(sm->getRobot(0));
    if (robot == nullptr) {
      std::fprintf(stderr, "[bench] robot 0 is not a FeatherstoneRobot\n");
      std::exit(4);
    }
    mb = robot->getDynamics()->getMultiBody();
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
  }

  void apply(const std::vector<double>& a) {
    for (size_t i = 0; i < thrusters.size(); ++i) thrusters[i]->setSetpoint(a[i]);
    for (size_t j = 0; j < servos.size(); ++j) servos[j]->setDesiredPosition(a[thrusters.size() + j]);
  }

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

  int action_dim() const { return int(thrusters.size() + servos.size()); }
};

// 平滑随机动作：推进器 OU 过程，限幅 [-amp, amp]；舵机目标在关节限位内缓慢游走。
struct RandomPolicy {
  std::mt19937 rng;
  std::normal_distribution<double> n01{0.0, 1.0};
  std::vector<double> a;
  int nt, ns;
  double amp;
  // SF_SAFE_TARGETS=1: 舵机目标留在关节限位以内（手指 Joint5/6 限位 [0,1]，目标取 [0.05,0.95]）。
  bool safe = std::getenv("SF_SAFE_TARGETS") != nullptr;
  // SF_SERVO_LIMITS="lo:hi,lo:hi,..."（按舵机顺序，由 servo_limits.py 生成）：目标在限位内缩进 10% 游走。
  std::vector<std::pair<double, double>> lim;
  RandomPolicy(int nt_, int ns_, unsigned seed, double amp_ = 0.4) : rng(seed), a(nt_ + ns_, 0.0), nt(nt_), ns(ns_), amp(amp_) {
    if (const char* env = std::getenv("SF_SERVO_LIMITS")) {
      std::string str(env);
      size_t pos = 0;
      while (pos < str.size()) {
        size_t comma = str.find(',', pos);
        std::string item = str.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        size_t colon = item.find(':');
        lim.emplace_back(std::atof(item.substr(0, colon).c_str()), std::atof(item.substr(colon + 1).c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
      if (int(lim.size()) != ns) {
        std::fprintf(stderr, "[bench] SF_SERVO_LIMITS has %zu entries, robot has %d servos\n", lim.size(), ns);
        std::exit(5);
      }
      for (int j = 0; j < ns; ++j) a[nt + j] = 0.5 * (lim[j].first + lim[j].second);
    }
    if (safe)
      for (int j = 4; j < ns; ++j) a[nt + j] = 0.5;
  }
  const std::vector<double>& next() {
    for (int i = 0; i < nt; ++i) a[i] = std::clamp(0.9 * a[i] + 0.15 * n01(rng), -amp, amp);
    if (!lim.empty()) {
      for (int j = 0; j < ns; ++j) {
        const double r = lim[j].second - lim[j].first;
        a[nt + j] = std::clamp(a[nt + j] + 0.03 * r * n01(rng), lim[j].first + 0.1 * r, lim[j].second - 0.1 * r);
      }
      return a;
    }
    for (int j = 0; j < ns; ++j) {
      const bool finger = j >= 4;
      const double lo = safe && finger ? 0.05 : -0.8, hi = safe && finger ? 0.95 : 0.8;
      a[nt + j] = std::clamp(0.95 * a[nt + j] + 0.05 * n01(rng), lo, hi);
    }
    return a;
  }
};

static double rot_angle(const btTransform& a, const btTransform& b) {
  btQuaternion d = a.getRotation().inverse() * b.getRotation();
  double w = std::min(1.0, std::fabs(double(d.getW())));
  return 2.0 * std::acos(w);
}

struct Err {
  double pos = 0, ori = 0, v = 0, w = 0, q = 0, qd = 0;
  void max_with(const State& a, const State& b) {
    pos = std::max(pos, double((a.base.getOrigin() - b.base.getOrigin()).length()));
    ori = std::max(ori, rot_angle(a.base, b.base));
    v = std::max(v, double((a.v - b.v).length()));
    w = std::max(w, double((a.w - b.w).length()));
    for (size_t k = 0; k < a.q.size(); ++k) {
      q = std::max(q, std::fabs(a.q[k] - b.q[k]));
      qd = std::max(qd, std::fabs(a.qd[k] - b.qd[k]));
    }
  }
  void print(const char* tag) const {
    std::printf("  %-34s pos %.3e m | ori %.3e rad | v %.3e m/s | w %.3e rad/s | q %.3e rad | qd %.3e rad/s\n", tag, pos,
                ori, v, w, q, qd);
  }
};

static int cmd_throughput(Sim& sim, double sim_seconds, int episode_steps, unsigned seed) {
  RandomPolicy pol(int(sim.thrusters.size()), int(sim.servos.size()), seed);
  const State init = sim.get();
  const int ctrl_steps = int(sim_seconds / (sim.fixed_dt * sim.decimation));
  double t_reset = 0.0;
  int resets = 0;
  const double t_sim0 = sim.sm->getSimulationTime();
  auto t0 = Clock::now();
  for (int t = 0; t < ctrl_steps; ++t) {
    sim.apply(pol.next());
    sim.step_ctrl();
    volatile State s = sim.get();  // 每步读取完整状态（训练时会写入回放池）
    (void)s;
    if (episode_steps > 0 && (t + 1) % episode_steps == 0) {
      auto tr = Clock::now();
      sim.set(init);
      t_reset += seconds_since(tr);
      ++resets;
    }
  }
  const double wall = seconds_since(t0);
  const double sim_t = sim.sm->getSimulationTime() - t_sim0;
  const double expected = ctrl_steps * sim.decimation * sim.fixed_dt;
  std::printf(
      "RESULT throughput phys_hz=%.0f decim=%d ctrl_steps=%d wall=%.3fs ctrl_steps_per_s=%.1f realtime_factor=%.1f "
      "resets=%d reset_us=%.2f sim_time=%.6f expected=%.6f\n",
      1.0 / sim.fixed_dt, sim.decimation, ctrl_steps, wall, ctrl_steps / wall, sim_t / wall, resets,
      resets ? 1e6 * t_reset / resets : 0.0, sim_t, expected);
  return 0;
}

static std::vector<State> rollout(Sim& sim, const std::vector<std::vector<double>>& actions, size_t from, size_t n) {
  std::vector<State> out;
  out.reserve(n);
  for (size_t t = from; t < from + n; ++t) {
    sim.apply(actions[t]);
    sim.step_ctrl();
    out.push_back(sim.get());
  }
  return out;
}

static int cmd_reset(Sim& sim) {
  const int T = 600, H = 100;
  RandomPolicy pol(int(sim.thrusters.size()), int(sim.servos.size()), 123);
  std::vector<std::vector<double>> actions;
  for (int t = 0; t < T + H; ++t) actions.push_back(pol.next());

  std::printf("robot: %zu thrusters, %zu servos, %zu dof links, decimation=%d, fixed_dt=%.6f\n", sim.thrusters.size(),
              sim.servos.size(), sim.dof_links.size(), sim.decimation, sim.fixed_dt);

  // 参考轨迹：states[t] 是执行 actions[t-1] 之后的状态，states[0] 为初始状态。
  std::vector<State> states{sim.get()};
  auto ref = rollout(sim, actions, 0, T + H - 1);
  states.insert(states.end(), ref.begin(), ref.end());
  std::printf("reference: final pos (%.3f %.3f %.3f), |v|=%.3f m/s\n", double(states.back().base.getOrigin().x()),
              double(states.back().base.getOrigin().y()), double(states.back().base.getOrigin().z()),
              double(states.back().v.length()));

  // 1) 重置到中途状态 s_k，用同样的动作重放 H 步，与参考轨迹逐步比较。
  std::printf("[1] reset to s_k, replay %d ctrl steps, max error vs reference:\n", H);
  for (int k : {0, 100, 250, 500}) {
    auto tr = Clock::now();
    sim.set(states[k]);
    const double us = 1e6 * seconds_since(tr);
    auto replay = rollout(sim, actions, k, H);
    Err e;
    for (int i = 0; i < H; ++i) e.max_with(replay[i], states[k + 1 + i]);
    char tag[64];
    std::snprintf(tag, sizeof(tag), "k=%d (set %.1f us)", k, us);
    e.print(tag);
  }

  // 2) 同一状态重置两次、同样动作 -> 是否逐位一致（确定性）。
  sim.set(states[250]);
  auto r1 = rollout(sim, actions, 250, H);
  sim.set(states[250]);
  auto r2 = rollout(sim, actions, 250, H);
  Err e2;
  for (int i = 0; i < H; ++i) e2.max_with(r1[i], r2[i]);
  std::printf("[2] determinism (same reset, same actions, twice):\n");
  e2.print("replay #1 vs replay #2");

  // 3) 重置到一个“人为构造”的状态（不在任何轨迹上）：位姿 + 速度 + 关节，读回检查。
  State custom = states[0];
  custom.base.setOrigin(btVector3(1.0, -2.0, 7.5));
  custom.base.setRotation(btQuaternion(btVector3(0.3, -0.2, 0.9).normalized(), 0.7));
  custom.v = btVector3(0.3, -0.1, 0.05);
  custom.w = btVector3(0.05, 0.1, -0.2);
  for (size_t k = 0; k < custom.q.size(); ++k) {
    custom.q[k] = 0.3 * std::sin(1.0 + k);
    custom.qd[k] = 0.1;
  }
  sim.set(custom);
  Err e3;
  e3.max_with(sim.get(), custom);
  std::printf("[3] set arbitrary state, read back immediately:\n");
  e3.print("readback error");
  sim.apply(std::vector<double>(sim.action_dim(), 0.0));
  sim.step_ctrl();
  State after = sim.get();
  std::printf("    after one zero-action ctrl step: pos (%.4f %.4f %.4f), v (%.4f %.4f %.4f)\n",
              double(after.base.getOrigin().x()), double(after.base.getOrigin().y()), double(after.base.getOrigin().z()),
              double(after.v.x()), double(after.v.y()), double(after.v.z()));
  return 0;
}

// 重置后误差随步数的增长曲线：区分“重置瞬间就有跳变”与“微小差异被动力学放大”。
static int cmd_resetprof(Sim& sim) {
  const int k = 100, H = 100;
  RandomPolicy pol(int(sim.thrusters.size()), int(sim.servos.size()), 123);
  std::vector<std::vector<double>> actions;
  for (int t = 0; t < k + H; ++t) actions.push_back(pol.next());
  std::vector<State> states{sim.get()};
  auto ref = rollout(sim, actions, 0, k + H);  // states[0..k+H]
  states.insert(states.end(), ref.begin(), ref.end());
  std::printf("reference max |qd| per joint:");
  for (size_t j = 0; j < states[0].qd.size(); ++j) {
    double m = 0;
    for (auto& st : states) m = std::max(m, std::fabs(st.qd[j]));
    std::printf(" %.2f", m);
  }
  std::printf(" rad/s\n");
  sim.set(states[k]);
  auto replay = rollout(sim, actions, k, H);
  std::printf("error after n ctrl steps from reset at k=%d (safe_targets=%d):\n", k, int(pol.safe));
  for (int n : {1, 2, 3, 5, 10, 20, 50, 100}) {
    Err e;
    e.max_with(replay[n - 1], states[k + n]);
    char tag[32];
    std::snprintf(tag, sizeof(tag), "n=%d", n);
    e.print(tag);
  }
  return 0;
}

// 阶跃响应：纵荡（4 个水平推进器）与垂荡（4 个垂直推进器），设定值 0.5，持续 10 s。
static int cmd_step(Sim& sim) {
  const State init = sim.get();
  struct Pattern {
    const char* name;
    std::vector<double> sp;
    int axis;
  };
  // 符号由 thrust 诊断得到：FrontRight/FrontLeft 正设定值向后推，BackRight/BackLeft 向前推。
  std::vector<Pattern> pats = {{"surge", {-0.5, -0.5, 0.5, 0.5, 0, 0, 0, 0}, 0},
                               {"heave", {0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5}, 2}};
  for (auto& p : pats) {
    sim.set(init);
    std::vector<double> a(sim.action_dim(), 0.0);
    for (size_t i = 0; i < p.sp.size(); ++i) a[i] = p.sp[i];
    std::printf("RESULT step %-5s v_axis at t=", p.name);
    for (int t = 1; t <= 500; ++t) {
      sim.apply(a);
      sim.step_ctrl();
      if (t == 25 || t == 50 || t == 100 || t == 250 || t == 500) {
        State s = sim.get();
        std::printf(" %.1fs:%.3f", t * 0.02, double(s.v[p.axis]));
      }
    }
    double total = 0;
    for (auto* th : sim.thrusters) total += std::fabs(double(th->getThrust()));
    std::printf("  (sum |thrust| = %.1f N)\n", total);
  }
  return 0;
}

static int cmd_current(Sim& sim) {
  sf::Ocean* ocean = sim.sm->getOcean();
  auto* uni = ocean ? dynamic_cast<sf::Uniform*>(ocean->getCurrent(0)) : nullptr;
  if (uni == nullptr) {
    std::printf("no uniform current found\n");
    return 1;
  }
  // Ocean 构造时 currentsEnabled=false，库内没有任何地方调用 EnableCurrents()（只有 GUI 开关会改），
  // 场景里定义的洋流默认不生效，必须显式打开。
  ocean->EnableCurrents();
  const State init = sim.get();
  std::vector<double> zero(sim.action_dim(), 0.0);
  for (double cx : {0.0, 0.3, 0.6}) {
    sim.set(init);
    uni->setVelocity(sf::Vector3(cx, 0.0, 0.0));
    for (int t = 0; t < 500; ++t) {  // 10 s @ 50 Hz
      sim.apply(zero);
      sim.step_ctrl();
    }
    State s = sim.get();
    std::printf("RESULT current_x=%.2f m/s -> after 10 s zero thrust: dx=%.3f m, vx=%.3f m/s, dz=%.3f m\n", cx,
                double(s.base.getOrigin().x() - init.base.getOrigin().x()), double(s.v.x()),
                double(s.base.getOrigin().z() - init.base.getOrigin().z()));
  }
  return 0;
}

// 诊断：零推力漂移（浮力是否配平），以及逐个推进器 +0.5 设定值 1 s 的响应。
static int cmd_thrust(Sim& sim) {
  const State init = sim.get();
  std::vector<double> a(sim.action_dim(), 0.0);
  for (size_t j = 0; j < sim.servos.size(); ++j) a[sim.thrusters.size() + j] = 0.0;
  sim.set(init);
  for (int t = 0; t < 250; ++t) {
    sim.apply(a);
    sim.step_ctrl();
  }
  State s = sim.get();
  std::printf("RESULT zero-thrust 5 s: dpos (%.3f %.3f %.3f) m, v (%.3f %.3f %.3f) m/s\n",
              double(s.base.getOrigin().x() - init.base.getOrigin().x()),
              double(s.base.getOrigin().y() - init.base.getOrigin().y()),
              double(s.base.getOrigin().z() - init.base.getOrigin().z()), double(s.v.x()), double(s.v.y()),
              double(s.v.z()));
  for (size_t i = 0; i < sim.thrusters.size(); ++i) {
    sim.set(init);
    std::fill(a.begin(), a.end(), 0.0);
    a[i] = 0.5;
    for (int t = 0; t < 50; ++t) {
      sim.apply(a);
      sim.step_ctrl();
    }
    s = sim.get();
    std::printf("RESULT %-15s sp=0.5 thrust=%7.3f N omega=%7.2f | 1 s: v (%6.3f %6.3f %6.3f) w (%6.3f %6.3f %6.3f)\n",
                sim.thrusters[i]->getName().c_str(), double(sim.thrusters[i]->getThrust()),
                double(sim.thrusters[i]->getOmega()), double(s.v.x()), double(s.v.y()), double(s.v.z()),
                double(s.w.x()), double(s.w.y()), double(s.w.z()));
  }
  return 0;
}

static int cmd_rebuild(Sim& sim) {
  const int n = 5;
  auto t0 = Clock::now();
  for (int i = 0; i < n; ++i) {
    sim.sm->RestartScenario();
    sim.sm->StartSimulation();
    sim.bind();
  }
  std::printf("RESULT rebuild RestartScenario+StartSimulation avg=%.1f ms\n", 1e3 * seconds_since(t0) / n);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <throughput|reset|current|rebuild> <scenario.scn> <data_dir> [phys_hz] [ctrl_hz] ...\n",
                 argv[0]);
    return 1;
  }
  const std::string cmd = argv[1], scn = argv[2], data = argv[3];
  const double phys_hz = argc > 4 ? std::atof(argv[4]) : 500.0;
  const double ctrl_hz = argc > 5 ? std::atof(argv[5]) : 50.0;

  Sim sim;
  auto t0 = Clock::now();
  sim.init(scn, data, phys_hz, ctrl_hz);
  std::printf("INIT %.1f ms (%zu thrusters, %zu servos, action_dim=%d)\n", 1e3 * seconds_since(t0), sim.thrusters.size(),
              sim.servos.size(), sim.action_dim());

  if (cmd == "throughput") {
    const double sim_seconds = argc > 6 ? std::atof(argv[6]) : 120.0;
    const int episode_steps = argc > 7 ? std::atoi(argv[7]) : 500;
    const unsigned seed = argc > 8 ? unsigned(std::atoi(argv[8])) : 0u;
    return cmd_throughput(sim, sim_seconds, episode_steps, seed);
  }
  if (cmd == "reset") return cmd_reset(sim);
  if (cmd == "current") return cmd_current(sim);
  if (cmd == "rebuild") return cmd_rebuild(sim);
  if (cmd == "thrust") return cmd_thrust(sim);
  if (cmd == "resetprof") return cmd_resetprof(sim);
  if (cmd == "step") return cmd_step(sim);
  std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
  return 1;
}
