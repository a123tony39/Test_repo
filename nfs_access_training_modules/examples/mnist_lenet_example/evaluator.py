import os
import time
import logging
from abc import ABC, abstractmethod

import training_utils
from orchid.computingEngine.utils import update_task_state, get_machine_ip

logging.basicConfig(
    format='%(levelname)s | %(asctime)s | %(module)s:%(lineno)d | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('Evaluating Log')
logger.setLevel(logging.INFO)

class Result():
    def __init__(self, result):
        self.metrics = result

class EvaluatorBase(ABC):
    def __init__(self, task):
        self.task = task

    def eval(self):
        result = self.evaluating_function(self.task)
        result = Result(result)  # ray.get use result.metrics to get result
        return result

    @abstractmethod
    def evaluating_function(self):
        pass

class Evaluator(EvaluatorBase):
    def set_parameters(self, config: dict):
        # set evaluator parameters
        self.cid = config.get('cid')
        self.datasetPath = config.get('datasetPath')
        self.localModelPath = config.get('localModelPath')
        self.taskname = config.get('taskname')
        self.model_name = config.get('modelName', None)
        self.datasetName = config.get('datasetName')
        self.exporter_url = config.get('exporter_url')
        self.device_name = config.get('device')
        self.machine_ip = get_machine_ip()
        # set hyperparameters
        hyperparams = config.get('hyperparams')
        self.batch_size = hyperparams.get('batchsize', 64)
        self.transform = hyperparams.get('transform', 'transforms.ToTensor()')
        self.epochs = hyperparams.get('epochs', 0)
        self.lr = hyperparams.get('learningrate', 1e-3)
        self.optimizer_name = hyperparams.get('optimizer', 'Adam')
        self.loss_func_name = hyperparams.get('lossFunction', 'MSE')
        
    def set_device(self):
        self.device = training_utils.setDevice(self.device_name)

    def set_dataloader(self):
        _, self.test_loader = training_utils.setDataLoader(
            cid = self.cid, 
            batch_size =  self.batch_size,
            datasetPath = self.datasetPath,
            datasetName = self.datasetName,
            transform = self.transform,
        )
    
    def set_model(self):
        # create model
        self.checkpoint_epochs, model = training_utils.setModel(
            cid = self.cid, 
            localModelPath = self.localModelPath,
            model_name = self.model_name,
        )
        self.model = model.to(self.device)

    def set_lossfunction(self):
        self.loss_function = training_utils.setLossfunc(
            loss_func_name=self.loss_func_name
        )

    def set_optimizer(self):
        self.optimizer = training_utils.setOptimizer(
            model=self.model,
            optimizer_name=self.optimizer_name,
            lr=self.lr
        )

    def evaluating_function(self, config: dict):
        self.set_parameters(config)
        self.set_device()
        self.set_dataloader()
        self.set_model()
        self.set_lossfunction()
        self.set_optimizer()
        # create model dir
        parentDir = os.path.dirname(self.localModelPath)
        os.makedirs(parentDir, exist_ok = True)
        # update exporter's task state 
        update_task_state(cid = self.cid, taskname = self.taskname, state = "RUNNING", machine_ip = self.machine_ip, exporter_url = self.exporter_url)
        # start evaluating
        logger.info("Client {} \033[1;34mstart evaluating\033[0m by {} now".format(self.cid, self.device))
        start_time = time.time()
        accuracy, loss = training_utils.model_evaluate(self.model, self.test_loader, self.loss_function, self.device)
        end_time = time.time()
        duration = end_time - start_time
        return {"accuracy":accuracy, "loss": loss, "epoch": self.checkpoint_epochs, "duration":duration, "machine_ip": self.machine_ip}