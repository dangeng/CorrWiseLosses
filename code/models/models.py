import torch

def create_model(opt):
    if opt.model == 'simple':
        from .simple_model import SimpleModel, InferenceModel
        if opt.isTrain:
            model = SimpleModel()
        else:
            model = InferenceModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
