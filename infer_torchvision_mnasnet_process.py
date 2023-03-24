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

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.dataset = param_map["dataset"]
        self.input_size = int(param_map["input_size"])
        self.model_path = param_map["model_path"]
        self.classes_path = param_map["classes_path"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {"dataset": self.dataset,
                     "input_size": str(self.input_size),
                     "model_path": self.model_path,
                     "classes_path": self.classes_path}
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class Mnasnet(dataprocess.CClassificationTask):

    def __init__(self, name, param):
        dataprocess.CClassificationTask.__init__(self, name)
        self.model = None
        self.colors = None
        self.class_names = []
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create parameters class
        if param is None:
            self.set_param_object(MnasnetParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def crop_image(self, src, width, height, box):
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])

        if x < 0:
            x = 0
        if x + w >= width:
            w = width - x - 1
        if y < 0:
            y = 0
        if x + h >= height:
            h = height - y - 1

        if w < 2 or h < 2:
            return None

        crop_img = src[y:y + h, x:x + w]
        return crop_img

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
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Step progress bar:
        self.emit_step_progress()

        # Load model
        if self.model is None or param.update:
            # Load class names
            self.read_class_names(param.classes_path)
            # Load model
            use_torchvision = param.dataset != "Custom"
            self.model = models.mnasnet(use_pretrained=use_torchvision,
                                          classes=len(self.class_names))
            if param.dataset == "Custom":
                self.model.load_state_dict(torch.load(param.model_path, map_location=self.device))

            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]
            self.model.to(self.device)
            param.update = False

        if not self.is_whole_image_classification():
            input_objects = self.get_input_objects()
            for obj in input_objects:
                roi_img = self.get_object_sub_image(obj)
                if roi_img is None:
                    continue

                predictions = self.predict(roi_img, param.input_size)
                class_index = predictions.argmax().item()
                self.add_object(obj, class_index, predictions[class_index].item())

        else:
            image_in = self.get_input(0)
            src_image = image_in.get_image()
            predictions = self.predict(src_image, param.input_size)
            sorted_data = sorted(zip(predictions.flatten().tolist(), self.get_names()), reverse=True)
            confidences = [str(conf) for conf, _ in sorted_data]
            names = [name for _, name in sorted_data]
            self.set_whole_image_results(names, confidences)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class MnasnetFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_torchvision_mnasnet"
        self.info.short_description = "MnasNet inference model for image classification."
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
        self.info.documentation_link = "https://arxiv.org/abs/1807.11626"
        self.info.repository = "https://github.com/pytorch/vision"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.icon_path = "icons/pytorch-logo.png"
        self.info.version = "1.3.0"
        self.info.keywords = "mnasnet,mobile,classification,cnn"

    def create(self, param=None):
        # Create process object
        return Mnasnet(self.info.name, param)
