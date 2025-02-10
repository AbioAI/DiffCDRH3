import yaml
import importlib

def get_obj_from_str(string, reload=False):
    #动态加载
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

# def instantiate_model(config):
#     """
#     从配置中加载模型类并实例化它
#     config: dict，包含模型的 `target` 和 `params`
#     """
#     # 获取目标模型的路径（模块名.类名）
#     target = config['target']
#
#     # 获取模块名和类名
#     module_name, class_name = target.rsplit('.', 1)
#
#     # 动态导入模块
#     module = importlib.import_module(module_name)
#
#     # 获取类
#     model_class = getattr(module, class_name)
#
#     # 获取参数
#     params = config.get('params', {})
#
#     # 实例化模型类
#     model = model_class(**params)
#
#     return model


def instantiate_from_config(config):
    # 从 config 中提取 first_stage_config 和 cond_stage_config
    first_stage_config = config.get("first_stage_config")
    cond_stage_config = config.get("cond_stage_config")
    diff_stage_config = config.get("diff_stage_config")

    # 动态实例化 first_stage 和 cond_stage
    first_stage = get_obj_from_str(first_stage_config["target"])(**first_stage_config.get("params", {}))
    cond_stage = get_obj_from_str(cond_stage_config["target"])(**cond_stage_config.get("params", {}))
    diff_stage = get_obj_from_str(diff_stage_config["target"])(**diff_stage_config.get("params", {}))
    # 返回实例化的对象
    return first_stage, cond_stage, diff_stage
