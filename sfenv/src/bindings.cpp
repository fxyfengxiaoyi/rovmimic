// sfenv：Stonefish 无界面仿真的 Python 绑定。
//
//   sim = sfenv.Sim(scenario, data_dir, phys_hz=100, ctrl_hz=50)
//   x = sim.get_state()            # body 向量 (nx,)，见 sf_sim.hpp
//   sim.set_state(x)
//   sim.step(a)                    # a = [推进器设定值 (nt), 舵机目标 (ns)]，推进一个控制步
//   X = sim.rollout(x0, A)         # 从 x0 出发执行 A (T, na)，返回每步之后的状态 (T, nx)
//   Y = sim.step_batch(X, A)       # 分时复用：对 N 个环境各推进一步，(N, nx) -> (N, nx)
//
// 一个进程只能创建一个 Sim（Stonefish 的 SimulationApp 是单例）。

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sf_sim.hpp"

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

bool g_created = false;

class PySim {
 public:
  PySim(const std::string& scn, const std::string& data, double phys_hz, double ctrl_hz) {
    if (g_created) throw std::runtime_error("only one sfenv.Sim per process (Stonefish SimulationApp is a singleton)");
    g_created = true;
    sim_.init(scn, data, phys_hz, ctrl_hz);
  }

  std::vector<std::string> thruster_names() const {
    std::vector<std::string> out;
    for (auto* t : sim_.thrusters) out.push_back(t->getName());
    return out;
  }
  std::vector<std::string> servo_names() const {
    std::vector<std::string> out;
    for (auto* s : sim_.servos) out.push_back(s->getName());
    return out;
  }
  std::vector<std::string> servo_joints() const {
    std::vector<std::string> out;
    for (auto* s : sim_.servos) out.push_back(s->getJointName());
    return out;
  }
  std::vector<std::string> link_names() const {
    std::vector<std::string> out;
    for (int i = 0; i < sim_.nlinks(); ++i) out.push_back(sim_.link_name(i));
    return out;
  }

  Array base_cg2o() const {
    Array out(7);
    double* o = out.mutable_data();
    const btQuaternion r = sim_.base_cg2o.getRotation();
    o[0] = sim_.base_cg2o.getOrigin().x(); o[1] = sim_.base_cg2o.getOrigin().y(); o[2] = sim_.base_cg2o.getOrigin().z();
    o[3] = r.getW(); o[4] = r.getX(); o[5] = r.getY(); o[6] = r.getZ();
    return out;
  }

  Array get_state() const {
    Array out(sim_.nx());
    sim_.get_body(out.mutable_data());
    return out;
  }

  void set_state(const Array& x) {
    check_1d(x, sim_.nx(), "state");
    sim_.set_body(x.data());
  }

  void step(const Array& a) {
    check_1d(a, sim_.na(), "action");
    py::gil_scoped_release nogil;
    sim_.apply(a.data());
    sim_.step_ctrl();
  }

  py::tuple rollout(const Array& x0, const Array& actions, bool return_links) {
    check_1d(x0, sim_.nx(), "x0");
    const int T = check_2d(actions, sim_.na(), "actions");
    const int nx = sim_.nx(), na = sim_.na(), nl = sim_.nlinks();
    Array states({T, nx});
    Array links(return_links ? std::vector<py::ssize_t>{T, nl, 7} : std::vector<py::ssize_t>{0, nl, 7});
    {
      py::gil_scoped_release nogil;
      sim_.set_body(x0.data());
      const double* a = actions.data();
      double* xs = states.mutable_data();
      double* ls = links.mutable_data();
      for (int t = 0; t < T; ++t) {
        sim_.apply(a + size_t(t) * na);
        sim_.step_ctrl();
        sim_.get_body(xs + size_t(t) * nx);
        if (return_links) sim_.link_poses(ls + size_t(t) * nl * 7);
      }
    }
    return py::make_tuple(states, links);
  }

  Array step_batch(const Array& xs, const Array& actions) {
    const int N = check_2d(xs, sim_.nx(), "states");
    if (check_2d(actions, sim_.na(), "actions") != N) throw std::invalid_argument("states/actions batch size mismatch");
    const int nx = sim_.nx(), na = sim_.na();
    Array out({N, nx});
    {
      py::gil_scoped_release nogil;
      for (int i = 0; i < N; ++i) {
        sim_.set_body(xs.data() + size_t(i) * nx);
        sim_.apply(actions.data() + size_t(i) * na);
        sim_.step_ctrl();
        sim_.get_body(out.mutable_data() + size_t(i) * nx);
      }
    }
    return out;
  }

  Array link_poses() const {
    Array out({sim_.nlinks(), 7});
    sim_.link_poses(out.mutable_data());
    return out;
  }

  sfsim::Sim sim_;

 private:
  static void check_1d(const Array& a, int n, const char* what) {
    if (a.ndim() != 1 || a.shape(0) != n)
      throw std::invalid_argument(std::string(what) + " must have shape (" + std::to_string(n) + ",)");
  }
  static int check_2d(const Array& a, int n, const char* what) {
    if (a.ndim() != 2 || a.shape(1) != n)
      throw std::invalid_argument(std::string(what) + " must have shape (N, " + std::to_string(n) + ")");
    return int(a.shape(0));
  }
};

}  // namespace

PYBIND11_MODULE(sfenv, m) {
  m.doc() = "Headless Stonefish simulation with exact state get/set (one Sim per process).";
  py::class_<PySim>(m, "Sim")
      .def(py::init<const std::string&, const std::string&, double, double>(), py::arg("scenario"),
           py::arg("data_dir"), py::arg("phys_hz") = 100.0, py::arg("ctrl_hz") = 50.0)
      .def_property_readonly("thruster_names", &PySim::thruster_names)
      .def_property_readonly("servo_names", &PySim::servo_names)
      .def_property_readonly("servo_joints", &PySim::servo_joints, "joint driven by each servo")
      .def_property_readonly("joint_names", [](const PySim& s) { return s.sim_.dof_names; },
                             "moving joints, in the order of q/qd in the state vector")
      .def_property_readonly("link_names", &PySim::link_names)
      .def_property_readonly("nt", [](const PySim& s) { return s.sim_.nt(); })
      .def_property_readonly("ns", [](const PySim& s) { return s.sim_.ns(); })
      .def_property_readonly("nq", [](const PySim& s) { return s.sim_.nq(); })
      .def_property_readonly("na", [](const PySim& s) { return s.sim_.na(); })
      .def_property_readonly("nx", [](const PySim& s) { return s.sim_.nx(); })
      .def_property_readonly("ctrl_dt", [](const PySim& s) { return s.sim_.ctrl_dt(); })
      .def_property_readonly("phys_dt", [](const PySim& s) { return s.sim_.fixed_dt; })
      .def_property_readonly("decimation", [](const PySim& s) { return s.sim_.decimation; })
      .def_property_readonly("sim_time", [](const PySim& s) { return double(s.sim_.sm->getSimulationTime()); })
      .def_property_readonly("base_cg2o", &PySim::base_cg2o, "base CG frame -> base_link origin frame (pos, quat wxyz)")
      .def("get_state", &PySim::get_state)
      .def("set_state", &PySim::set_state, py::arg("x"))
      .def("step", &PySim::step, py::arg("action"))
      .def("rollout", &PySim::rollout, py::arg("x0"), py::arg("actions"), py::arg("return_links") = false,
           "returns (states (T, nx), link_poses (T, nlinks, 7) or empty)")
      .def("step_batch", &PySim::step_batch, py::arg("states"), py::arg("actions"))
      .def("link_poses", &PySim::link_poses);
}
