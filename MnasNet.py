from ikomia import dataprocess
import MnasNet_process as processMod
import MnasNet_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class MnasNet(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.MnasNetProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.MnasNetWidgetFactory()
