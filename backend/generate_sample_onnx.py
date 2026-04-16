"""
generate_sample_onnx.py
Creates a minimal dummy YOLOv8-shaped ONNX model that outputs random detections.
Use this as a placeholder until you upload your real model.

Run: python generate_sample_onnx.py
Requires: pip install onnx numpy
"""
import numpy as np
import os

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    # YOLOv8 output: [1, 5, 8400] — cx,cy,w,h,conf per anchor
    N_ANCHORS = 8400

    # Input node
    X = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 640, 640])

    # Output node  
    Y = helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1, 5, N_ANCHORS])

    # Random weight initializer (simulates detections scattered around center)
    np.random.seed(42)
    data = np.zeros((1, 5, N_ANCHORS), dtype=np.float32)
    # Plant some fake pothole detections
    for i in range(5):
        idx = np.random.randint(0, N_ANCHORS)
        data[0, 0, idx] = np.random.uniform(200, 440)   # cx
        data[0, 1, idx] = np.random.uniform(200, 400)   # cy
        data[0, 2, idx] = np.random.uniform(60, 150)    # w
        data[0, 3, idx] = np.random.uniform(40, 100)    # h
        data[0, 4, idx] = np.random.uniform(0.50, 0.92) # conf

    initializer = numpy_helper.from_array(data, name='dummy_output')

    # Constant node that returns the initializer
    const_node = helper.make_node('Constant', inputs=[], outputs=['output0'],
                                  value=numpy_helper.from_array(data))

    graph = helper.make_graph([const_node], 'pothole_dummy', [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8

    os.makedirs('../models', exist_ok=True)
    out_path = '../models/pothole_dummy.onnx'
    onnx.save(model, out_path)
    print(f'✅ Dummy ONNX model saved → {out_path}')
    print('   Upload this via the UI or pass as model_path in app.py')

except ImportError:
    print('onnx package not found. Install with: pip install onnx')
    print('Alternatively, just run the app without a model — demo mode works fine.')
