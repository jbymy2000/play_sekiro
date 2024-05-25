from pysekiro.on_policy import Play_Sekiro_Online
target = 'boss1' # 苇名弦一郎

train = Play_Sekiro_Online(
    save_memory_path = target + '_memory.json',    # 注释这行就不保存记忆
    load_memory_path = target + '_memory.json',    # 注释这行就不加载记忆
    save_weights_path = target + '_w.h5',    # 注释这行就不保存模型权重
    load_weights_path = target + '_w.h5'     # 注释这行就不加载模型权重
)
train.run()
