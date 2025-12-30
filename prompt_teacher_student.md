我们现在将使用目前实现的记忆功能充分观测的world model作为teacher policy, 训练一个观测不足的student policy。我会将流程拆分为几步，请你分步实现，每实现一步之后要进行训练验证,验证无误后，提交commit，再进行下一个点。实现代码时，不能影响已有的代码机构，尽量进行增量的设计，创建另外的代码。
1. 两个policy架构一致，同时推理，teacher全部冻结，student可训练部分与train gen only阶段相同。
2. 训练过程中，同一组数据输入给两个policy，但是student policy无法获取head camera观测作为输入，只有wrist camera。
3. student policy逼近teacher policy的KV memory state, 用memory的差异作为一个loss, 与pred出的下一帧图像的gt_loss进行加权
4. 实现对照组，student policy仅使用gt_loss。