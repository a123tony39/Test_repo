import os
import ray
import time
import torch
import pickle
import logging
import numpy as np
from abc import ABC, abstractmethod
from ray import train
from ray.air import session, Checkpoint
from ray.air.config import ScalingConfig
from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig
from ray.train.torch import prepare_data_loader, prepare_model

from orchid.computingEngine.ray.train import TorchTrainer

logging.basicConfig(
    format='%(levelname)s | %(asctime)s | %(module)s:%(lineno)d | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('Training Log')
logger.setLevel(logging.INFO)

DEVICE_CPU = "CPU"
DEVICE_GPU = "GPU"
DEVICE_RASPBERRY_PI_4 = "RP4"
DEVICE_JESTON = "Jetson"

class TrainerBase(ABC):
    def __init__(self, task, log_dir, log_dir_name='training_result'):
        # parameters
        num_workers = task['numWorkers']
        stop=eval(task['stop'])
        # get the cluster resourecs
        resources = ray.cluster_resources()
        # use which device
        if task['device'] == DEVICE_GPU:
            if "GPU" not in resources:
                raise ValueError("There is no enough GPU to train")
            else:
                resources_per_worker = {}
                use_gpu = True # will make resources_per_worker add the resource {"GPU":1}
        elif task['device'] == DEVICE_RASPBERRY_PI_4:
            if "RP4" not in resources:
                raise ValueError("There is no enough RP4 to train")
            else:
                resources_per_worker = {"CPU":1, "RP4":1}
                use_gpu = False
        else:
            resources_per_worker = {"CPU":1}
            use_gpu = False
        # define scaling and run configs
        scaling_config = ScalingConfig(
            num_workers = num_workers, 
            use_gpu = use_gpu,
            resources_per_worker = resources_per_worker,
        )
        run_config = RunConfig(
            name = log_dir_name,
            local_dir = log_dir,
            checkpoint_config = CheckpointConfig(num_to_keep=1),
            stop=stop,
        )
        # trainer
        self.ray_trainer = TorchTrainer(
            train_loop_per_worker = self.training_function,
            train_loop_config = task,
            scaling_config = scaling_config,
            run_config = run_config,
        )

    @abstractmethod
    def training_function(self, config: dict):
        pass

    def fit(self):
        result = self.ray_trainer.fit()
        return result
    

class Trainer(TrainerBase):
    def training_function(self, config: dict):
        import training_utils
        from orchid.computingEngine.utils import update_task_state, get_machine_ip
        # device, dataset and model path
        cid = config.get('cid')
        datasetPath = config.get('datasetPath')
        localModelPath = config.get('localModelPath')
        taskname = config.get('taskname')
        model_name = config.get('modelName', None)
        datasetName = config.get('datasetName')
        exporter_url = config.get('exporter_url')
        # update task state
        machine_ip = get_machine_ip()
        update_task_state(cid = cid, taskname = taskname, state = "RUNNING", machine_ip = machine_ip, exporter_url = exporter_url)
        # hyperparams
        hyperparams = config.get('hyperparams')
        batch_size = hyperparams.get('batchsize', 64)
        transform = hyperparams.get('transform', 'transforms.ToTensor()')
        epochs = hyperparams.get('epochs', 0)
        lr = hyperparams.get('learningrate', 1e-3)
        optimizer_name = hyperparams.get('optimizer', 'Adam')
        loss_func_name = hyperparams.get('lossFunction', 'MSE')
        round_num = config.get('roundNum')
        seeds_for_epochs = config.get('seedsForEpochs')
        # set device
        device = train.torch.get_device()
        # dataloader
        batch_size_per_worker = batch_size // session.get_world_size()
        train_loader, test_loader = training_utils.setDataLoader(
            cid = cid, 
            batch_size = batch_size_per_worker,
            datasetPath = datasetPath,
            datasetName = datasetName,
            transform = transform,
        )
        train_loader = prepare_data_loader(train_loader)
        test_loader = prepare_data_loader(test_loader)
        # create model
        checkpoint_epochs, model = training_utils.setModel(
            cid = cid, 
            localModelPath = localModelPath,
            model_name = model_name,
        )
        # optimizer and loss function
        loss_function = training_utils.setLossfunc(
            loss_func_name=loss_func_name
        )
        optimizer = training_utils.setOptimizer(
            model=model,
            optimizer_name=optimizer_name,
            lr=lr
        )
        # prepared model
        model = prepare_model(model)
        # create model dir
        parentDir = os.path.dirname(localModelPath)
        os.makedirs(parentDir, exist_ok = True)
        # record the start time
        start_time = time.time()
        logger.info("Client {} \033[1;34mstart training\033[0m by {} now".format(cid, device))
        logger.info("Seeds for epochs: {}".format(seeds_for_epochs))
        # print("Client {} \033[1;34mstart training\033[0m by {} now".format(cid, device))
        epoch = 0
        for epoch in range(epochs):
            # model training
            logger.info("Client {} \033[1;34mstart training\033[0m {} epoch".format(cid, epoch))
            seed = seeds_for_epochs[(round_num-1) * epochs + epoch] if isinstance(seeds_for_epochs, list) else None
            training_utils.train_loop_per_worker(train_loader, model, loss_function, optimizer, device, seed)
            # print("epoch: {} accuracy: {} loss: {}".format(epoch, accuracy, loss))
            logger.info("Client {} \033[1;34mfinish training\033[0m {} epoch".format(cid, epoch))
            checkpoint=Checkpoint.from_dict(dict(epochs=checkpoint_epochs+epoch+1, model_state_dict=model.state_dict()))
            checkpoint_path = checkpoint.to_directory(parentDir)
            # print(checkpoint_path)
            # # Evaluate every epoch
            # accuracy, loss = model_evaluate(model, test_loader, loss_function, device)
            # end_time = time.time()
            # duration = end_time - start_time
            # session.report({"accuracy":accuracy, "loss": loss, "epoch": epoch, "duration":duration})
        accuracy, loss = training_utils.model_evaluate(model, test_loader, loss_function, device)
        end_time = time.time()
        duration = end_time - start_time
        session.report({"accuracy":accuracy, "loss": loss, "epoch": epoch, "duration":duration, "machine_ip":machine_ip})
        if epochs>0: self.save_as_npy(model.state_dict(), localModelPath)

    def save_as_npy(self, model_state_dict, npy_path):
        model_ndarrays = np.array([ndarr.cpu().numpy() for ndarr in model_state_dict.values()], dtype=object)
        np.save(npy_path, model_ndarrays, allow_pickle=True, fix_imports=True)


class YoloV5Trainer(TrainerBase):
    def training_function(self, config: dict):
        # self modules
        # import torch_utils
        from orchid.computingEngine.utils import update_task_state, get_machine_ip
        from yolov5 import train as yolov5_train
        # device, dataset and model path
        cid = config.get('cid')
        datasetPath = config.get('datasetPath')
        localModelPath = config.get('localModelPath')
        taskname = config.get('taskname')
        model_name = config.get('modelName')
        datasetName = config.get('datasetName')
        # update task state
        machine_ip = get_machine_ip()
        update_task_state(cid = cid, taskname = taskname, state = "RUNNING", machine_ip = machine_ip)
        # hyperparams
        hyperparams = config.get('hyperparams')
        batch_size = hyperparams.get('batchsize', 64)
        transform = hyperparams.get('transform', 'transforms.ToTensor()')
        epochs = hyperparams.get('epochs', 0)
        lr = hyperparams.get('learningrate', 1e-3)
        optimizer_name = hyperparams.get('optimizer', 'Adam')
        loss_func_name = hyperparams.get('lossFunction', 'MSE')
        # set device
        device = train.torch.get_device()
        # # dataloader
        # batch_size_per_worker = batch_size // session.get_world_size()
        # train_loader, test_loader = torch_utils.setDataLoader(
        #     cid = cid, 
        #     batch_size = batch_size_per_worker,
        #     datasetPath = datasetPath,
        #     datasetName = datasetName,
        #     transform = transform,
        # )
        # train_loader = prepare_data_loader(train_loader)
        # test_loader = prepare_data_loader(test_loader)
        # # create model
        # checkpoint_epochs, model = torch_utils.setModel(
        #     cid = cid, 
        #     localModelPath = localModelPath,
        #     model_name = model_name,
        # )
        # # optimizer and loss function
        # loss_function = torch_utils.setLossfunc(
        #     loss_func_name=loss_func_name
        # )
        # optimizer = torch_utils.setOptimizer(
        #     model=model,
        #     optimizer_name=optimizer_name,
        #     lr=lr
        # )
        # # prepared model
        # model = prepare_model(model)
        # create model dir
        parentDir = os.path.dirname(localModelPath)
        os.makedirs(parentDir, exist_ok = True)
        # record the start time
        start_time = time.time()
        print("Client {} \033[1;34mstart training\033[0m by {} now".format(cid, device))

        # yolov5 : will save npy model in yolov5_train.run()
        data_yaml_path = os.path.join(os.path.dirname(parentDir), 'data.yaml')
        model_ndarray = np.load(localModelPath[:-3]+'.npy', allow_pickle=True)
        results = yolov5_train.run(model_ndarray, data=data_yaml_path, 
                imgsz=512, epochs=epochs,
                weights='/opt/nfsfl/modules_yolov5/yolov5/runs/train/exp/weights/best.pt')
        
        metrics_path = os.path.join(os.path.dirname(localModelPath), 'metrics.npy')
        metrics_ndarray = np.load(metrics_path, allow_pickle=True)
        p, r, map50, map50_95, lbox, lobj, lcls = metrics_ndarray[0]
        # yolov5 end

        # epoch = 0
        # for epoch in range(epochs):
        #     # model training
        #     torch_utils.train_loop_per_worker(train_loader, model, loss_function, optimizer, device)
        #     # print("epoch: {} accuracy: {} loss: {}".format(epoch, accuracy, loss))
        #     checkpoint=Checkpoint.from_dict(dict(epochs=checkpoint_epochs+epoch+1, model_state_dict=model.state_dict()))
        #     checkpoint_path = checkpoint.to_directory(parentDir)
        #     # print(checkpoint_path)
        #     # # Evaluate every epoch
        #     # accuracy, loss = model_evaluate(model, test_loader, loss_function, device)
        #     # end_time = time.time()
        #     # duration = end_time - start_time
        #     # session.report({"accuracy":accuracy, "loss": loss, "epoch": epoch, "duration":duration})
        # accuracy, loss = torch_utils.model_evaluate(model, test_loader, loss_function, device)
        end_time = time.time()
        duration = end_time - start_time
        session.report({"accuracy":map50, "loss": lcls, "p": p, "r": r, "map50": map50, "map": map,
                        "epoch": epochs, "duration":duration, "machine_ip":machine_ip})
        # if epochs>0: self.ray_checkpoint_to_torch(checkpoint_path, localModelPath)

    def ray_checkpoint_to_torch(self, ray_checkpoint_path, localModelPath):
        # change the model format from ray .pkl to torch
        ray_checkpoint_path = os.path.join(ray_checkpoint_path, 'dict_checkpoint.pkl')
        with open(ray_checkpoint_path, 'rb') as f:
            model = pickle.load(f)
        model_state_dict = model['model_state_dict']
        new_dict = {}
        for key, value in model_state_dict.items():
            new_key = key.replace("module.", "")
            new_dict[new_key] = value
        model['model_state_dict'] = new_dict
        torch.save(model, localModelPath)