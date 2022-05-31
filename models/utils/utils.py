import jittor as jt


def start_grad(model):
    for param in model.parameters():
        param.start_grad()


def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)

def forgiving_state_restore(net, loaded_dict):
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict:
            new_loaded_dict[k] = loaded_dict[k]
            print(k, "match")
        elif k.replace("original.", "") in loaded_dict:
            new_loaded_dict[k] = loaded_dict[k.replace("original.", "")]
            print(k, "match after adjustment")
        else:
            print(k, 'unmatch')
            # logging.info("Skipped loading parameter %s", k)
    # net_state_dict.update(new_loaded_dict)
    net.load_state_dict(new_loaded_dict)
    return net
