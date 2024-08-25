import os

from src.models.silicone_mask import SiliconeMaskModel

base_path = '../../weights'

model = SiliconeMaskModel('../../ckpt/best_model.keras')

model.export_pb(os.path.join(base_path, '/saved_model'))
model.export_onnx(base_path)