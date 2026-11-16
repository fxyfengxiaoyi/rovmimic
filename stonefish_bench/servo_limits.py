"""按舵机在场景中的出现顺序打印其关节限位，格式 "lo:hi,lo:hi,..."，供 sf_bench 的 SF_SERVO_LIMITS 使用。"""

import sys
import xml.etree.ElementTree as ET

robot = ET.parse(sys.argv[1]).getroot().find("robot")
limits = {}
for j in robot.findall("joint"):
  lim = j.find("limit")
  if lim is not None:
    limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
out = []
for a in robot.findall("actuator"):
  if a.get("type") == "servo":
    lo, hi = limits[a.find("joint").get("name")]
    out.append(f"{lo}:{hi}")
print(",".join(out))
