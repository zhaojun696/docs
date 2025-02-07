#%%
import pkg_resources

def load_plugins():
    for entry_point in pkg_resources.iter_entry_points('mkdocs.plugins'):
        # plugin_class = entry_point.load()
        # plugin_instance = plugin_class()
        # 使用插件实例
        print(f'Loaded plugin: {entry_point.name}')

load_plugins()