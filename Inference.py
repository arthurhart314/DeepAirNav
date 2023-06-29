import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

class TRTModel():
    def __init__(self, batch_size, ONNX_FILE_PATH):
        self.ONNX_FILE_PATH = ONNX_FILE_PATH
        self.batch_size = batch_size
        self.output_tensor = None
        pass

    def build_engine(self):
        model_path = self.ONNX_FILE_PATH
        TRT_LOGGER = trt.Logger()
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as builder_config, trt.OnnxParser(network, TRT_LOGGER) as parser:

            if builder.platform_has_fast_fp16:
                builder_config.set_flag(trt.BuilderFlag.FP16)

            with open(model_path, "rb") as f:
                parser.parse(f.read())
            
            self.engine = builder.build_engine(network, builder_config)

            self.context = self.engine.create_execution_context()

    def reserve_cuda_memory(self):
        engine, context = self.build_engine()

        for binding in engine:
            if engine.binding_is_input(binding):  # we expect only one input
                #self.device_input_tensor = torch.tensor(np.zeros([self.batch_size , 3, 128, 128], dtype = np.float32), device = torch.device('cuda'))
                pass
            else: 
                self.output_tensor = torch.tensor(np.zeros([self.batch_size , 128], dtype = np.float32), device = torch.device('cuda'))

        self.stream = cuda.Stream()
	
    def run(self, images_tensor):
        self.context.execute_async_v2(bindings=[int(images_tensor.data_ptr()), 
                                                int(self.device_output_tensor.data_ptr())], 
                                      stream_handle=self.stream.handle)
        self.stream.synchronize()

        return self.output_tensor