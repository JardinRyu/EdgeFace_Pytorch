# config.py

cfg_EdgeFace = {
    #'name': 'EdgeFacenet',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 50,
    'decay2': 68,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 64,
    'out_channel': 64
}