import timm


def model_factory(num_classes: int, pretrained: str):
    model = timm.create_model(pretrained, pretrained=True, num_classes=num_classes, in_chans=1)

    default_size = model.default_cfg["input_size"][-1]

    return model, default_size
