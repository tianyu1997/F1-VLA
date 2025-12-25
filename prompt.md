我们现在将在f1_vla大模型的基础，加入记忆功能。大致思路参考可/mnt/data2/ty/F1-VLA/ME_KVM_VLA/src下面的文档，但是注意这些代码可能有bug，不能完全参考。我会将流程拆分为几步，请你分步实现，每实现一步之后要进行训练验证,验证无误后，提交commit，再进行下一个点。
流程：
1. config中加入一个记忆开关，如果关闭则流程与原来一致，如果打开则进行如下修改
2. 数据读取：sequential读取，以episode为单位，episode内frame需要按顺序读取，episode可以打乱；batch数据要包含dataset_index, episode_index, frame_index。
3. 要将action和state以及其历史信息（长度、index与image_history对齐）以文本模式输入给paligemma。
4. memory设计：
    a. 需要维护一个memory bank, 根据数据index读取上一时刻的memory，即previous_memory。
    b. 学习一个init_memory(nn.parameters),若frame_index==0，则使用previous_memory=init_memory。
    c. previous_memory设计为kv_cache的形式，在模型推理时拼接到kv_cache。
    d. 学习一个memory token, 输入给paligemma之后，对应的输出部分则为本时刻的memory_info。
    e. memory_info和previous_memory经过gru模型，得到current_memory,存进memory bank。
5. 采用bptt方式memory梯度detach，长度在config中配置。