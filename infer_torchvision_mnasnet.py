from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from infer_torchvision_mnasnet.infer_torchvision_mnasnet_process import MnasnetFactory
        # Instantiate process object
        return MnasnetFactory()

    def getWidgetFactory(self):
        from infer_torchvision_mnasnet.infer_torchvision_mnasnet_widget import MnasnetWidgetFactory
        # Instantiate associated widget object
        return MnasnetWidgetFactory()
