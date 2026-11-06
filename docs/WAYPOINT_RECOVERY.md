# 每个途经点的恢复轮次

`motion_recovery.max_recovery_rounds=3`：每个途经点共最多3轮恢复，两轴共用节点轮号。正常首次目标命令不占恢复轮数；首次实际写出恢复指令才进入第1轮。

一轮可包含多次升降档和0.2/0.6/1.2°三个偏置档。`upshifts`、`downshifts`、`angle_changes`只是本轮调节步骤的计数，不分别消耗恢复轮数。完整一轮仍失败并达到无效观察窗上限后，实际写回原名义目标才开始下一轮；写入失败不计入下一轮。最多3轮后结束，不能无限重复。

换途经点重置轮次、步骤计数和恢复截止时间，每点仍受24秒绝对恢复期限限制；整条路径另有有限超时。恢复轮内的小幅进展不重置次数或无限延长期限。普通升档速度下限跨同一单调段保留；90只用于当前节点最后一级，新节点继承普通60下限，避免在90档叠加新偏置。重启下一轮也先将90召回普通上限、写回原目标再调角。

诊断增加`waypoint_index`、`recovery_round`、`max_recovery_rounds`、`total_rounds_started`和`recovery_budget_scope=waypoint`，事件携带节点和轮号。

离线测试：

```bash
python3 -m unittest discover -s tests -p 'test_waypoint_retries.py' -v
```

测试运行真实move与A6打包，驱动、时间和反馈由虚拟对象替换，没有打开串口。开发板替换源码不自动启动机械动作。
