import os
from tqdm import tqdm
import numpy as np


def calculate_weigths_labels(dataloader, root_dir):
    # Create an instance from the data loader
    z = np.zeros((2,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        mask = sample['label'].detach().cpu().view(-1).numpy()
        labels = np.where(mask > 0, 1, 0).astype(np.uint8)
        count_l = np.bincount(labels, minlength=2)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(root_dir, 'train_classes_weights.npy')
    np.save(classes_weights_path, ret)

    return ret


def model_info(model, logger, report='summary'):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report == 'full':
        logger.info('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    logger.info('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))
