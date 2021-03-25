from ikomia import core, dataprocess
from ikomia.dnn.torch import models
import os
import copy
# Your imports below
import cv2
import torch
import torchvision.transforms as transforms


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class MnasNetParam(core.CProtocolTaskParam):

    def __init__(self):
        core.CProtocolTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = 'mnasnet'
        self.dataset = 'ImageNet'
        self.input_size = 224
        self.model_path = ''
        self.classes_path = os.path.dirname(os.path.realpath(__file__)) + "/models/imagenet_classes.txt"
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.dataset = param_map["dataset"]
        self.input_size = int(param_map["input_size"])
        self.model_path = param_map["model_path"]
        self.classes_path = param_map["classes_path"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["dataset"] = self.dataset
        param_map["input_size"] = str(self.dataset)
        param_map["model_path"] = self.model_path
        param_map["classes_path"] = self.classes_path
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class MnasNetProcess(dataprocess.CImageProcess2d):

    def __init__(self, name, param):
        dataprocess.CImageProcess2d.__init__(self, name)
        self.model = None
        self.class_names = []
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Remove graphics input
        self.removeInput(1)
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CDblFeatureIO())

        # Create parameters class
        if param is None:
            self.setParam(MnasNetParam())
        else:
            self.setParam(copy.deepcopy(param))

    def load_class_names(self):
        self.class_names.clear()
        param = self.getParam()

        with open(param.classes_path) as f:
            for row in f:
                self.class_names.append(row[:-1])

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def predict(self, image, input_size):
        input_img = cv2.resize(image, (input_size, input_size))

        trs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        input_tensor = trs(input_img)
        input_tensor = input_tensor.unsqueeze(0)
        prob = None

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)

        return prob

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get parameters :
        param = self.getParam()

        # Get input :
        input = self.getInput(0)
        src_image = input.getImage()

        h = src_image.shape[0]
        w = src_image.shape[1]

        # Step progress bar:
        self.emitStepProgress()

        # Load model
        if self.model is None or param.update:
            # Load class names
            self.load_class_names()
            # Load model
            use_torchvision = param.dataset != "Custom"
            self.model = models.mnasnet(use_pretrained=use_torchvision,
                                          classes=len(self.class_names))
            if param.dataset == "Custom":
                self.model.load_state_dict(torch.load(param.model_path))

            param.update = False

        pred = self.predict(src_image, param.input_size)

        # Step progress bar:
        self.emitStepProgress()

        # Forward input image
        self.forwardInputImage(0, 0)

        # Set graphics output
        graphics_output = self.getOutput(1)
        graphics_output.setNewLayer("MnasNet")
        graphics_output.setImageIndex(0)
        class_index = pred.argmax()
        msg = self.class_names[class_index] + ": {:.3f}".format(pred[class_index])
        graphics_output.addText(msg, 0.05 * w, 0.05 * h)

        # Init numeric output
        numeric_ouput = self.getOutput(2)
        numeric_ouput.clearData()
        numeric_ouput.setOutputType(dataprocess.NumericOutputType.TABLE)
        numeric_ouput.addValueList(pred.flatten().tolist(), "Probability", self.class_names)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class MnasNetProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "MnasNet"
        self.info.shortDescription = "MnasNet inference model for image classification."
        self.info.description = "MnasNet inference model for image classification. " \
                                "Implementation from PyTorch torchvision package. " \
                                "This Ikomia plugin can make inference of pre-trained model from " \
                                "ImageNet dataset or custom trained model. Custom training can be made with " \
                                "the associated MnasNetTrain plugin from Ikomia marketplace."
        self.info.authors = "Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le"
        self.info.article = "MnasNet: Platform-Aware Neural Architecture Search for Mobile"
        self.info.journal = "Conference on Computer Vision and Pattern Recognition (CVPR)"
        self.info.year = 2019
        self.info.licence = "BSD-3-Clause License"
        self.info.documentationLink = "https://arxiv.org/abs/1807.11626"
        self.info.repository = "https://github.com/pytorch/vision"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.iconPath = "icons/pytorch-logo.png"
        self.info.version = "1.0.1"
        self.info.keywords = "mnasnet,mobile,classification,cnn"

    def create(self, param=None):
        # Create process object
        return MnasNetProcess(self.info.name, param)
