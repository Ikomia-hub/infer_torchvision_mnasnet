from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class MnasNet(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from MnasNet.MnasNet_process import MnasNetProcessFactory
        # Instantiate process object
        return MnasNetProcessFactory()

    def getWidgetFactory(self):
        from MnasNet.MnasNet_widget import MnasNetWidgetFactory
        # Instantiate associated widget object
        return MnasNetWidgetFactory()
