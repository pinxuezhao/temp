import abc
import importlib
import os
import warnings
from typing import Dict

from pipe.models.simple_partitioning_config import PipelineConfig

_PARTITIONED_MODELS_PACKAGE = "models.partitioned"


class CommonModelHandler(abc.ABC):
    def __init__(self, partitioned_models_package=_PARTITIONED_MODELS_PACKAGE):
        self.partitioned_models_package = partitioned_models_package
        self.generated_file_name_or_path = None
        self.normal_model_instance = None
        self.generated = None
        self.pipe_config = None

    @abc.abstractmethod
    def _get_normal_model_instance(self, *args, **kw):
        raise NotImplementedError()

    def get_normal_model_instance(self, *args, **kw):
        if self.normal_model_instance is None:
            self.normal_model_instance = self._get_normal_model_instance(*args, **kw)
        return self.normal_model_instance

    def get_generated_module(self):
        if self.generated is None:
            cfg = self.generated_file_name_or_path

            is_full_path = os.path.exists(cfg)
            try:
                if is_full_path:
                    generated = load_module(cfg)
                else:
                    generated_file_name = self.generated_file_name_or_path
                    generated = importlib.import_module("." + generated_file_name,
                                                        package=self.partitioned_models_package)
            except Exception as e:
                print(f"-E- error loading generated config given {cfg}. is_full_path={is_full_path}")
                raise e

            self.generated = generated

        return self.generated

    def get_pipe_config(self) -> PipelineConfig:
        if self.pipe_config is None:
            generated = self.get_generated_module()
            GET_PARTITIONS_ON_CPU = True
            create_pipeline_configuration = generated.create_pipeline_configuration
            config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)
            pipe_config = PipelineConfig(config)
            self.pipe_config = pipe_config
        return self.pipe_config

    def realize_stage_for_rank(self, batch_size, my_rank):
        pipe_config = self.get_pipe_config()
        layers, tensors = self.get_layers_and_tensors()
        return pipe_config.realize_stage_for_rank(layers, tensors, batch_size, my_rank)

    def get_layers_and_tensors(self, *args, **kw):
        if self.normal_model_instance is None:
            self.normal_model_instance = self.get_normal_model_instance()
        model_instance = self.normal_model_instance
        pipe_config = self.get_pipe_config()
        generated = self.get_generated_module()
        layerDict = generated.layerDict
        tensorDict = generated.tensorDict
        depth = pipe_config.d['depth']
        blocks = pipe_config.d['basic_blocks']
        layers = layerDict(model_instance, depth=depth, basic_blocks=blocks)
        tensors = tensorDict(model_instance)
        return layers, tensors

    def get_loader(self, *args, **kw):
        raise NotImplementedError()

    def get_extra(self, *args, **kwargs):
        """extra keywords for dataset,
        return a dict if there is something to return"""
        pass

    def register_autogenerated(self, generated_file_name_or_path: str):
        self.generated_file_name_or_path = generated_file_name_or_path
        register_model(generated_file_name_or_path=generated_file_name_or_path, handler=self)

    def set_partitioned_models_package(self, partitioned_models_package):
        self.partitioned_models_package = partitioned_models_package


AVAILABLE_MODELS: Dict[str, CommonModelHandler] = {}


def register_model(generated_file_name_or_path, handler: CommonModelHandler):
    global AVAILABLE_MODELS
    AVAILABLE_MODELS[generated_file_name_or_path] = handler


def register_model_func(generated_file_name_or_path, _get_normal_model_instance, get_extra=None):
    d = dict(_get_normal_model_instance=_get_normal_model_instance)
    if get_extra:
        d['get_extra'] = get_extra
    handler_cls = type("AutoGeneratedModelHandler",
                       (CommonModelHandler,),
                       d
                       )
    handler: CommonModelHandler = handler_cls()
    handler.register_autogenerated(generated_file_name_or_path=generated_file_name_or_path)


def load_module(full_path: str):
    # "/path/to/file.py"
    spec = importlib.util.spec_from_file_location("module.name", full_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo


# TODO: register normal model by function.
# create handler class (see examples in dir)
# register the handler.

NORMAL_MODEL_ENTRY_POINTS = {}
NORMAL_MODEL_ENTRY_POINTS_HANDLERS = {}


def register_normal_model_by_function(fn):
    # Register ad normal model
    model_name = fn.__name__
    NORMAL_MODEL_ENTRY_POINTS[model_name] = fn

    # create entry point handler
    class EntryPointFunctionModelHandler(CommonModelHandler):
        def __init__(self, normal_model_fn, *args, **kw):
            super().__init__(*args, **kw)
            self.normal_model_fn = normal_model_fn

        def _get_normal_model_instance(self, *args, **kwargs):
            return self.normal_model_fn(*args, **kwargs)

    handler = EntryPointFunctionModelHandler(normal_model_fn=fn)
    if model_name in NORMAL_MODEL_ENTRY_POINTS_HANDLERS:
        warnings.warn(f"model_name {model_name} already exisits in NORMAL_MODEL_ENTRY_POINTS_HANDLERS")
    NORMAL_MODEL_ENTRY_POINTS_HANDLERS[model_name] = handler


def normal_model_entry_point(model_name):
    return NORMAL_MODEL_ENTRY_POINTS[model_name]


def normal_model_entry_point_handler(model_name):
    return NORMAL_MODEL_ENTRY_POINTS_HANDLERS[model_name]
