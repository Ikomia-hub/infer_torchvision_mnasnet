from ikomia import core, dataprocess
from ikomia.dnn.torch import models
import os
import copy
import cv2
import torch
import torchvision.transforms as transforms
import random


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class MnasnetParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
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
class Mnasnet(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.model = None
        self.colors = None
        self.class_names = []
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CDataStringIO())

        # Create parameters class
        if param is None:
            self.setParam(MnasnetParam())
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
        return 2

    def predict(self, image, input_size):
        input_img = cv2.resize(image, (input_size, input_size))

        trs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        input_tensor = trs(input_img).to(self.device)
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
        image_in = self.getInput(0)
        src_image = image_in.getImage()
        graphics_in = self.getInput(1)

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

            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]
            self.model.to(self.device)
            param.update = False

        # Prepare outputs
        graphics_output = self.getOutput(1)
        graphics_output.setNewLayer("ResNet")
        graphics_output.setImageIndex(0)
        table_output = self.getOutput(2)
        table_output.setOutputType(dataprocess.NumericOutputType.TABLE)
        table_output.clearData()

        objects_to_classify = []
        if graphics_in.isDataAvailable():
            for item in graphics_in.getItems():
                if not item.isTextItem():
                    objects_to_classify.append(item)

        if len(objects_to_classify) > 0:
            names = []
            confidences = []
            boxes = []
            ids = []

            for obj in objects_to_classify:
                # Inference
                rc = obj.getBoundingRect()
                x = int(rc[0])
                y = int(rc[1])
                w = int(rc[2])
                h = int(rc[3])
                crop_img = src_image[y:y+h, x:x+w]
                pred = self.predict(crop_img, param.input_size)
                class_index = pred.argmax()
                # Box
                prop_rect = core.GraphicsRectProperty()
                prop_rect.pen_color = self.colors[class_index]
                graphics_box = graphics_output.addRectangle(rc[0], rc[1], rc[2], rc[3], prop_rect)
                graphics_box.setCategory(self.class_names[class_index])
                # Label
                msg = str(graphics_box.getId()) + " - " + self.class_names[class_index]
                prop_text = core.GraphicsTextProperty()
                prop_text.font_size = 10
                prop_text.color = self.colors[class_index]
                graphics_output.addText(msg, rc[0], rc[1], prop_text)
                # Result values
                names.append(self.class_names[class_index])
                ids.append(str(graphics_box.getId()))
                confidences.append(str(pred[class_index].item()))
                boxes.append("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(rc[0], rc[1], rc[2], rc[3]))

            # Results table output
            table_output.addValueList(confidences, "Confidence", names)
            table_output.addValueList(ids, "Graphics object")
            table_output.addValueList(boxes, "Boxes")
        else:
            pred = self.predict(src_image, param.input_size)
            # Set graphics output
            class_index = pred.argmax()
            msg = self.class_names[class_index] + ": {:.3f}".format(pred[class_index])
            graphics_output.addText(msg, 0.05 * w, 0.05 * h)
            # Set numeric output
            sorted_data = sorted(zip(pred.flatten().tolist(), self.class_names), reverse=True)
            confidences = [str(conf) for conf, _ in sorted_data]
            names = [name for _, name in sorted_data]
            table_output.addValueList(confidences, "Probability", names)

        # Step progress bar:
        self.emitStepProgress()

        # Forward input image
        self.forwardInputImage(0, 0)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class MnasnetFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_torchvision_mnasnet"
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
        return Mnasnet(self.info.name, param)
