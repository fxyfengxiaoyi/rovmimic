"""从 u0env 的 BlueROV2 场景（bluerov2_arm.scn / bluerov2_alpha5_robotiq.scn）生成无界面（console）基准场景。

生成五个变体：
  rov_nav.scn    : USIM 导航类任务（goto/scan/inspect/follow）实际用的 bluerov2.scn，无机械臂
  rov_only.scn   : bluerov2_arm.scn 去掉机械臂（压载/浮块和 rov_nav 不同，仅作基准对比）
  rov_arm.scn    : BlueROV2 Heavy + 仓库自带的 robot_arm（4 关节 + 2 指；舵机无限速，参数不现实）
  rov_alpha5.scn       : BlueROV2 Heavy + Alpha5 + Robotiq Hand-E，取自 U0 实际使用的
                         bluerov2_alpha5_robotiq.scn。Alpha5 臂段网格（Reach 授权文件，由 u0env 的
                         tools/dataprocess/pkg_install.py 生成到 data/alpha5/）齐全时与原模型完全一致；
                         缺失时退化为下面的圆柱代理。
  rov_alpha5_proxy.scn : 同上，但 5 个臂段始终换成圆柱体代理（保留原文件给定的质量与惯量值，长度由
                         惯量反推）。原文件没写 <cg>，Stonefish 的质心位置和惯量主轴方向由几何决定，
                         所以代理模型的臂段质心/主轴与原模型不同，仅用于对比。

处理：替换 $(arg ...)；删除所有传感器、灯光和 ROS 标签（训练直接读多体状态，
相机等视觉传感器在 console 模式下也无法工作）；外面包一层只有海水的环境
（无海底、无波浪、无粒子，一个可在运行时修改的 uniform 洋流）。
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovmimic.paths import U0ENV_DIR  # noqa: E402

U0ENV = U0ENV_DIR / "ros_ws/src/stonefish_bluerov2"
SRC = U0ENV / "scenarios/robots/bluerov2_arm.scn"
SRC_ALPHA5 = U0ENV / "scenarios/robots/bluerov2_alpha5_robotiq.scn"  # USIM 抓取/搬运类任务
SRC_NAV = U0ENV / "scenarios/robots/bluerov2.scn"  # USIM 导航类任务

# 缺失网格的 Alpha5 臂段 -> 圆柱代理 (radius, height)。height 由 sqrt(12*I/m) 估算。
ALPHA5_PROXY = {
  "alpha5/M3-INLINE.obj": (0.020, 0.154),
  "alpha5/M2-1-1.obj": (0.020, 0.040),
  "alpha5/M2.obj": (0.020, 0.158),
  "alpha5/M2-1-3.obj": (0.020, 0.040),
  "alpha5/RS1-100-101-123.obj": (0.020, 0.106),
}

ARGS = {"vehicle_name": "bluerov2", "position": "0.0 0.0 5.0", "orientation": "0.0 0.0 0.0"}

ARM_LINKS = {"ArmBaseLink", "ArmLink1", "ArmLink2", "ArmLink3", "ArmLink4", "EE", "Finger1", "Finger2"}

WORLD_HEADER = """<?xml version="1.0"?>
<scenario>
  <environment>
    <ned latitude="0.0" longitude="0.0"/>
    <ocean>
      <water density="1031.0" jerlov="0.15"/>
      <waves height="0.0"/>
      <particles enabled="false"/>
      <current type="uniform">
        <velocity xyz="0.0 0.0 0.0"/>
      </current>
    </ocean>
  </environment>
  <materials>
    <material name="Neutral" density="1000.0" restitution="0.1"/>
    <material name="Fiberglass" density="1500.0" restitution="0.3"/>
    <material name="Steel" density="7810.0" restitution="0.2"/>
    <material name="Dummy" density="1000.0" restitution="0.1"/>
    <friction_table>
      <friction material1="Neutral" material2="Neutral" static="0.5" dynamic="0.2"/>
    </friction_table>
  </materials>
  <looks>
    <look name="black" gray="0.05" roughness="0.2"/>
    <look name="Blue" rgb="0.0 0.3 1.0" roughness="0.3"/>
    <look name="None" gray="0.5" roughness="0.5"/>
  </looks>
"""


def load_robot_scenario(src: Path = SRC) -> ET.Element:
  text = src.read_text()
  for k, v in ARGS.items():
    text = text.replace(f"$(arg {k})", v)
  return ET.fromstring(text)


def strip_robot(robot: ET.Element, keep_arm: bool) -> None:
  for child in list(robot):
    tag = child.tag
    if tag in {"sensor", "light", "comm", "ros_subscriber", "ros_publisher"}:
      robot.remove(child)
    elif not keep_arm and tag == "link" and child.get("name") in ARM_LINKS:
      robot.remove(child)
    elif not keep_arm and tag == "joint":
      robot.remove(child)
    elif not keep_arm and tag == "actuator" and child.get("type") == "servo":
      robot.remove(child)


def proxy_alpha5_links(robot: ET.Element) -> int:
  """把引用缺失 Alpha5 网格的 model 连杆改成 cylinder，质量/惯量/材料保持不变。"""
  n = 0
  for link in robot.findall("link"):
    phys = link.find("physical")
    mesh = phys.find("mesh") if phys is not None else None
    if mesh is None or mesh.get("filename") not in ALPHA5_PROXY:
      continue
    radius, height = ALPHA5_PROXY[mesh.get("filename")]
    origin = phys.find("origin")
    idx = list(link).index(phys)
    link.remove(phys)
    vis = link.find("visual")
    if vis is not None:
      link.remove(vis)
    link.set("type", "cylinder")
    dims = ET.Element("dimensions", radius=str(radius), height=str(height))
    link.insert(idx, dims)
    link.insert(idx + 1, origin if origin is not None else ET.Element("origin", xyz="0 0 0", rpy="0 0 0"))
    n += 1
  return n


def alpha5_meshes_available() -> bool:
  return all((U0ENV / "data" / f).exists() for f in ALPHA5_PROXY)


def build(keep_arm: bool, src: Path = SRC, proxy: bool = False) -> str:
  root = load_robot_scenario(src)
  parts = []
  for looks in root.findall("looks"):
    parts.append(ET.tostring(looks, encoding="unicode"))
  robot = root.find("robot")
  strip_robot(robot, keep_arm)
  if proxy:
    assert proxy_alpha5_links(robot) == len(ALPHA5_PROXY)
  parts.append(ET.tostring(robot, encoding="unicode"))
  return WORLD_HEADER + "\n".join(parts) + "\n</scenario>\n"


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--out", default=str(Path(__file__).parent / "scenarios"))
  args = p.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  real = alpha5_meshes_available()
  print(f"Alpha5 meshes {'found' if real else 'MISSING -> rov_alpha5 uses cylinder proxies'}")
  variants = [("rov_nav", False, SRC_NAV, False), ("rov_only", False, SRC, False), ("rov_arm", True, SRC, False),
              ("rov_alpha5", True, SRC_ALPHA5, not real), ("rov_alpha5_proxy", True, SRC_ALPHA5, True)]
  for name, keep_arm, src, proxy in variants:
    path = out / f"{name}.scn"
    path.write_text(build(keep_arm, src, proxy))
    robot = ET.fromstring(path.read_text()).find("robot")
    counts = {}
    for c in robot:
      key = c.tag if c.tag != "actuator" else f"actuator:{c.get('type')}"
      counts[key] = counts.get(key, 0) + 1
    print(f"{path}: {counts}")


if __name__ == "__main__":
  main()
