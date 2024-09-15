def get_model_size(model):
    def convert_to_gigabytes(input_megabyte):
        gigabyte = 1.0 / 1024
        convert_gb = gigabyte * input_megabyte
        return convert_gb

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2

    print('model size: {:.3f} GB'.format(convert_to_gigabytes(size_all_mb)))
