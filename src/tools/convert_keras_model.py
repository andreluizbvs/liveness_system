import os

from src.models.liveness import LivenessModel

base_path = '../../weights'

model = LivenessModel('../../ckpt/best_model.keras')

model.export_pb(os.path.join(base_path, '/saved_model'))
model.export_onnx(base_path)